/**
 * 两个上游模型服务的客户端。
 * 「攻」侧是 GPT-SoVITS 的推理服务（默认 :9880），
 * 「防」侧是 AASIST 辨伪服务（默认 :9890，见部署包 README_DEPLOY.md）。
 *
 * TTS 服务在不同版本里同时支持 GET 查询串和 POST JSON 两种调用方式，
 * 这里先试 POST，失败再退回 GET，并把实际生效的方式记进返回值，方便排查。
 */
import net from 'node:net';
import fs from 'node:fs';
import path from 'node:path';
import { buildMultipart } from './multipart.mjs';

const AUDIO_MIME = {
    '.wav': 'audio/wav',
    '.flac': 'audio/flac',
    '.mp3': 'audio/mpeg',
    '.ogg': 'audio/ogg',
    '.m4a': 'audio/mp4',
};

export function mimeOf(file) {
    return AUDIO_MIME[path.extname(file).toLowerCase()] || 'application/octet-stream';
}

/** 只用 TCP 连接判断端口是否在监听，不依赖上游是否有健康检查路由 */
export function probePort(baseUrl, timeoutMs = 1200) {
    return new Promise((resolve) => {
        let url;
        try {
            url = new URL(baseUrl);
        } catch {
            resolve(false);
            return;
        }
        const port = Number(url.port || (url.protocol === 'https:' ? 443 : 80));
        const socket = net.connect({ host: url.hostname, port });
        const done = (ok) => {
            socket.destroy();
            resolve(ok);
        };
        socket.setTimeout(timeoutMs);
        socket.once('connect', () => done(true));
        socket.once('timeout', () => done(false));
        socket.once('error', () => done(false));
    });
}

async function fetchWithTimeout(url, options, timeoutMs) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        return await fetch(url, { ...options, signal: controller.signal });
    } finally {
        clearTimeout(timer);
    }
}

/** 辨伪服务的健康信息，拿到 device 之类的信息用于界面展示 */
export async function detectHealth(cfg) {
    const alive = await probePort(cfg.detect.baseUrl);
    if (!alive) return { up: false };
    try {
        const res = await fetchWithTimeout(
            new URL(cfg.detect.healthPath, cfg.detect.baseUrl),
            { method: 'GET' },
            4000,
        );
        const body = await res.json();
        return { up: true, ...body };
    } catch {
        return { up: true };
    }
}

/** 调 GPT-SoVITS 合成一段语音，返回音频字节 */
export async function synthesize(cfg, params) {
    const payload = { ...cfg.tts.defaults, ...params };
    const url = new URL(cfg.tts.path, cfg.tts.baseUrl);

    const attempts = [];
    let lastError = null;

    // 方式一：POST JSON
    try {
        const res = await fetchWithTimeout(
            url,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            },
            cfg.tts.timeoutMs,
        );
        if (res.ok) {
            const buf = Buffer.from(await res.arrayBuffer());
            if (buf.length > 1024) return { buffer: buf, via: 'POST /tts (json)', attempts };
            attempts.push(`POST 返回 ${buf.length} 字节，疑似不是音频`);
        } else {
            attempts.push(`POST 失败 HTTP ${res.status}`);
            lastError = new Error(await res.text().catch(() => `HTTP ${res.status}`));
        }
    } catch (err) {
        attempts.push(`POST 异常 ${err.message}`);
        lastError = err;
    }

    // 方式二：GET 查询串
    try {
        const qs = new URLSearchParams();
        for (const [k, v] of Object.entries(payload)) {
            if (v === null || v === undefined || v === '') continue;
            qs.set(k, typeof v === 'boolean' ? String(v).toLowerCase() : String(v));
        }
        const res = await fetchWithTimeout(`${url}?${qs}`, { method: 'GET' }, cfg.tts.timeoutMs);
        if (res.ok) {
            const buf = Buffer.from(await res.arrayBuffer());
            if (buf.length > 1024) return { buffer: buf, via: 'GET /tts (query)', attempts };
            attempts.push(`GET 返回 ${buf.length} 字节，疑似不是音频`);
        } else {
            attempts.push(`GET 失败 HTTP ${res.status}`);
        }
    } catch (err) {
        attempts.push(`GET 异常 ${err.message}`);
    }
    throw new Error(`合成服务调用失败：${attempts.join('；')}${lastError ? `（${lastError.message}）` : ''}`);
}

/** 把一段音频交给辨伪服务打分 */
export async function detectFile(cfg, filePath, filename = path.basename(filePath)) {
    const buffer = fs.readFileSync(filePath);
    return detectBuffer(cfg, buffer, filename);
}

export async function detectBuffer(cfg, buffer, filename = 'upload.wav', contentType = null) {
    const { boundary, body } = buildMultipart('file', filename, contentType || mimeOf(filename), buffer);
    const res = await fetchWithTimeout(
        new URL(cfg.detect.path, cfg.detect.baseUrl),
        {
            method: 'POST',
            headers: { 'Content-Type': `multipart/form-data; boundary=${boundary}` },
            body,
        },
        cfg.detect.timeoutMs,
    );
    const text = await res.text();
    let json;
    try {
        json = JSON.parse(text);
    } catch {
        throw new Error(`辨伪服务返回了非 JSON 内容（HTTP ${res.status}）：${text.slice(0, 200)}`);
    }
    if (!res.ok) {
        throw new Error(json?.error || `辨伪服务返回 HTTP ${res.status}`);
    }
    return json;
}

/** 按阈值把分数翻译成结论与对抗判定 */
export function judge(score, threshold) {
    const verdict = score >= threshold ? 'real' : 'fake';
    return { verdict, margin: Number((score - threshold).toFixed(6)) };
}
