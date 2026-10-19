/**
 * 语音攻防演示平台 —— 服务端。
 *
 * 只依赖 Node 内置模块，不需要 npm install，也不做构建。
 * 职责有三块：托管前端静态文件；把两类请求转发给上游模型服务；
 * 记录每一次对抗回合，供页面展示与导出。
 *
 * 端口默认 8790，仅监听本机。绑定域名的事以后再做。
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { loadConfig, ensureDirs, samplesDir, WEB_DIR, ROOT, WORKSPACE, REF_DIR } from './config.mjs';
import { addRun, clearRuns, listRuns, mediaPath, newId, refPath, statSafe } from './store.mjs';
import { detectBuffer, detectFile, detectHealth, judge, mimeOf, probePort, synthesize } from './upstream.mjs';
import { parseMultipart } from './multipart.mjs';

const cfg = loadConfig();
ensureDirs();

const SAMPLE_INFO = {
    'real.wav': { label: 'ASVspoof2019 LA 真人', kind: '真声', known: 5.285, expect: 'real' },
    'fake.wav': { label: 'ASVspoof2019 LA 伪造（A14）', kind: '训练分布内伪造', known: -2.817, expect: 'fake' },
    'clone.wav': { label: 'GPT-SoVITS v4 零样本克隆', kind: '跨代克隆', known: -0.791, expect: 'real' },
    'ref_real.wav': { label: '同说话人真声（clone 的 prompt）', kind: '真声', known: 2.089, expect: 'real' },
};

function json(res, code, payload) {
    const body = JSON.stringify(payload);
    res.writeHead(code, {
        'Content-Type': 'application/json; charset=utf-8',
        'Cache-Control': 'no-store',
    });
    res.end(body);
}

function sendFile(res, file, { download = false, contentType = null } = {}) {
    const st = statSafe(file);
    if (!st || !st.isFile()) {
        json(res, 404, { error: '文件不存在' });
        return;
    }
    const headers = {
        'Content-Type': contentType || mimeOf(file),
        'Content-Length': st.size,
        'Cache-Control': 'no-store',
    };
    if (download) headers['Content-Disposition'] = `attachment; filename="${encodeURIComponent(path.basename(file))}"`;
    res.writeHead(200, headers);
    fs.createReadStream(file).pipe(res);
}

async function readBody(req, limit = 80 * 1024 * 1024) {
    const chunks = [];
    let size = 0;
    for await (const chunk of req) {
        size += chunk.length;
        if (size > limit) throw new Error('请求体过大');
        chunks.push(chunk);
    }
    return Buffer.concat(chunks);
}

async function readJson(req) {
    const buf = await readBody(req);
    if (!buf.length) return {};
    return JSON.parse(buf.toString('utf8'));
}

/** 从请求里取出音频：既支持裸字节流（浏览器直接拖文件），也支持 multipart 表单 */
async function extractAudio(req, url) {
    const queryFile = url.searchParams.get('media');
    if (queryFile && queryFile.startsWith('sample:')) {
        const dir = samplesDir(cfg);
        const name = path.basename(queryFile.slice('sample:'.length));
        if (!dir) throw new Error('未找到辨伪部署包，无法读取样例');
        const file = path.join(dir, name);
        if (!statSafe(file)) throw new Error(`样例不存在：${name}`);
        return { kind: 'path', file, filename: name };
    }
    if (queryFile && queryFile.startsWith('media:')) {
        const id = path.basename(queryFile.slice('media:'.length));
        const file = mediaPath(id);
        if (!statSafe(file)) throw new Error(`本地音频不存在：${id}`);
        return { kind: 'path', file, filename: path.basename(file) };
    }

    const buf = await readBody(req);
    if (!buf.length) throw new Error('没有收到音频数据');
    const contentType = req.headers['content-type'] || '';
    if (contentType.startsWith('multipart/form-data')) {
        const { fields, files } = parseMultipart(buf, contentType);
        const first = Object.values(files)[0];
        if (!first) throw new Error('表单里没有文件字段');
        return {
            kind: 'buffer',
            buffer: first.data,
            filename: first.filename,
            contentType: first.contentType,
            fields,
        };
    }
    const filename = url.searchParams.get('filename') || 'upload.wav';
    return { kind: 'buffer', buffer: buf, filename, contentType: mimeOf(filename) };
}

async function runDetect(audio) {
    if (audio.kind === 'path') return detectFile(cfg, audio.file, audio.filename);
    return detectBuffer(cfg, audio.buffer, audio.filename, audio.contentType);
}

/**
 * 把前端传来的引用解析成绝对路径。
 * 上游 GPT-SoVITS 与本站同机运行，直接给本机路径即可。
 * 支持三种写法：sample:xxx.wav（辨伪包样例）、ref:xxx（上传的参考音）、以及本来就是绝对路径。
 */
function resolveRef(ref) {
    if (!ref) return null;
    if (ref.startsWith('sample:')) {
        const dir = samplesDir(cfg);
        if (!dir) throw new Error('未找到辨伪部署包，无法引用样例音频');
        return path.join(dir, path.basename(ref.slice('sample:'.length)));
    }
    if (ref.startsWith('ref:')) {
        const id = path.basename(ref.slice('ref:'.length));
        const hit = fs.readdirSync(REF_DIR).find((name) => name.replace(/\.[^.]+$/, '') === id);
        if (!hit) throw new Error(`参考音频不存在：${id}`);
        return path.join(REF_DIR, hit);
    }
    if (ref.startsWith('media:')) return mediaPath(path.basename(ref.slice('media:'.length)));
    if (ref.startsWith('ttsvoice:')) {
        const dir = cfg.modelDirs.gptsovits ? path.join(cfg.modelDirs.gptsovits, 'voices') : null;
        if (!dir) throw new Error('未找到合成部署包，无法引用其参考音频');
        return path.join(dir, path.basename(ref.slice('ttsvoice:'.length)));
    }
    return ref;
}

/** 组装一份送给合成服务的参数 */
function ttsParams(body) {
    const params = { ...body.params, text: (body.text || '').trim() };
    const ref = resolveRef(body.ref);
    if (ref) params.ref_audio_path = ref;
    else delete params.ref_audio_path;
    if (!params.prompt_text) delete params.prompt_text;
    return params;
}

function decorate(result) {
    const threshold = Number(result.threshold_aasist ?? cfg.detect.threshold);
    const verdict = result.aasist_verdict || judge(result.aasist_score, threshold).verdict;
    return {
        ...result,
        threshold_aasist: threshold,
        verdict,
        margin: Number((Number(result.aasist_score) - threshold).toFixed(6)),
        /** 判真说明克隆语音骗过了检测器，即「攻击成功」 */
        attacked: verdict === 'real',
    };
}

const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '::1', '[::1]']);

/**
 * 访问控制。令牌留空时完全不限制，适合只在本机用。
 * 一旦填了令牌，非本机 Host 的请求必须带令牌，三种方式任选：
 *   ?token=xxx（顺手下发 cookie）、X-Voice-Arena-Token 请求头、va_token cookie。
 * 本机 Host 一律放行，免得本地调试每次都要拼参数。
 */
function checkAccess(req, url) {
    const token = cfg.access?.token;
    if (!token) return { ok: true, setCookie: false };
    const host = String(req.headers.host || '').split(':')[0].toLowerCase();
    if (LOCAL_HOSTS.has(host)) return { ok: true, setCookie: false };
    if (url.searchParams.get('token') === token) return { ok: true, setCookie: true };
    if (req.headers['x-voice-arena-token'] === token) return { ok: true, setCookie: false };
    const cookie = req.headers.cookie || '';
    const hit = /(?:^|;\s*)va_token=([^;]+)/.exec(cookie);
    if (hit && decodeURIComponent(hit[1]) === token) return { ok: true, setCookie: false };
    return { ok: false, setCookie: false };
}

const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, `http://${req.headers.host || '127.0.0.1'}`);
    const route = url.pathname;
    try {
        const access = checkAccess(req, url);
        if (!access.ok) {
            if (route.startsWith('/api/')) {
                return json(res, 401, { error: '需要访问令牌，请在网址后加 ?token=<令牌>' });
            }
            res.writeHead(401, { 'Content-Type': 'text/html; charset=utf-8' });
            res.end(
                '<!doctype html><meta charset="utf-8"><title>401</title>' +
                    '<body style="font:15px/1.7 system-ui;background:#0e1418;color:#dce6ec;padding:48px">' +
                    '<h1 style="font-size:20px">需要访问令牌</h1>' +
                    '<p style="color:#8ba0ad">在网址后加 <code>?token=你的令牌</code> 再试一次，' +
                    '例如 <code>https://voice.soyorin.work/?token=xxx</code>。带过一次之后会记住 30 天。</p>',
            );
            return undefined;
        }
        if (access.setCookie) {
            res.setHeader(
                'Set-Cookie',
                `va_token=${encodeURIComponent(cfg.access.token)}; Path=/; Max-Age=2592000; SameSite=Lax`,
            );
        }
        if (route === '/api/state' && req.method === 'GET') {
            const [detect, ttsUp] = await Promise.all([detectHealth(cfg), probePort(cfg.tts.baseUrl)]);
            return json(res, 200, {
                app: { version: '0.1.0', root: ROOT, workspace: WORKSPACE, hostname: os.hostname() },
                tts: { up: ttsUp, baseUrl: cfg.tts.baseUrl, path: cfg.tts.path, defaults: cfg.tts.defaults },
                detect: { ...detect, baseUrl: cfg.detect.baseUrl, path: cfg.detect.path, threshold: cfg.detect.threshold },
                modelDirs: cfg.modelDirs,
                runs: listRuns().length,
            });
        }

        if (route === '/api/config' && req.method === 'GET') {
            return json(res, 200, cfg);
        }

        if (route === '/api/samples' && req.method === 'GET') {
            const dir = samplesDir(cfg);
            const items = [];
            if (dir && statSafe(dir)) {
                for (const name of fs.readdirSync(dir)) {
                    const st = statSafe(path.join(dir, name));
                    if (!st || !st.isFile() || !/\.(wav|flac|mp3)$/i.test(name)) continue;
                    items.push({
                        name,
                        bytes: st.size,
                        url: `/media/sample/${encodeURIComponent(name)}`,
                        ...(SAMPLE_INFO[name] || {}),
                    });
                }
            }
            // 合成包自带的参考音，附上同名 .txt 里的逐字文本，选中后可自动填进 prompt
            const voices = [];
            const voiceDir = cfg.modelDirs.gptsovits ? path.join(cfg.modelDirs.gptsovits, 'voices') : null;
            if (voiceDir && statSafe(voiceDir)) {
                for (const name of fs.readdirSync(voiceDir)) {
                    if (!/\.(wav|flac|mp3)$/i.test(name)) continue;
                    const file = path.join(voiceDir, name);
                    const st = statSafe(file);
                    const txt = file.replace(/\.[^.]+$/, '.txt');
                    const promptText = statSafe(txt) ? fs.readFileSync(txt, 'utf8').trim() : '';
                    voices.push({ name, bytes: st?.size ?? 0, promptText, url: `/media/voice/${encodeURIComponent(name)}` });
                }
            }
            return json(res, 200, { dir, items, voices, voiceDir });
        }

        if (route === '/api/runs' && req.method === 'GET') {
            return json(res, 200, { items: listRuns() });
        }

        if (route === '/api/runs' && req.method === 'DELETE') {
            clearRuns();
            return json(res, 200, { ok: true });
        }

        if (route === '/api/tts' && req.method === 'POST') {
            const body = await readJson(req);
            const text = (body.text || '').trim();
            if (!text) return json(res, 400, { error: '文本不能为空' });
            const params = ttsParams(body);
            const started = Date.now();
            const { buffer, via, attempts } = await synthesize(cfg, params);
            const id = newId('tts');
            const file = mediaPath(id, 'wav');
            fs.writeFileSync(file, buffer);
            const record = addRun({
                id,
                kind: 'tts',
                createdAt: new Date().toISOString(),
                text,
                params: { ...params, ref_audio_path: params.ref_audio_path || null },
                via,
                attempts,
                elapsedMs: Date.now() - started,
                audio: { id, url: `/media/${id}.wav`, bytes: buffer.length },
            });
            return json(res, 200, record);
        }

        if (route === '/api/detect' && req.method === 'POST') {
            const audio = await extractAudio(req, url);
            const started = Date.now();
            const raw = await runDetect(audio);
            const result = decorate(raw);
            const record = addRun({
                id: newId('det'),
                kind: 'detect',
                createdAt: new Date().toISOString(),
                source: audio.filename,
                elapsedMs: Date.now() - started,
                result,
            });
            return json(res, 200, record);
        }

        if (route === '/api/attack' && req.method === 'POST') {
            const body = await readJson(req);
            const text = (body.text || '').trim();
            if (!text) return json(res, 400, { error: '文本不能为空' });
            const t0 = Date.now();
            const params = ttsParams(body);
            const synth = await synthesize(cfg, params);
            const ttsMs = Date.now() - t0;
            const id = newId('atk');
            const file = mediaPath(id, 'wav');
            fs.writeFileSync(file, synth.buffer);
            const t1 = Date.now();
            const raw = await detectBuffer(cfg, synth.buffer, `${id}.wav`, 'audio/wav');
            const detectMs = Date.now() - t1;
            const result = decorate(raw);
            const record = addRun({
                id,
                kind: 'attack',
                createdAt: new Date().toISOString(),
                text,
                via: synth.via,
                attempts: synth.attempts,
                ttsMs,
                detectMs,
                audio: { id, url: `/media/${id}.wav`, bytes: synth.buffer.length },
                result,
                outcome: result.attacked ? 'attacked' : 'blocked',
            });
            return json(res, 200, record);
        }

        if (route === '/api/refs' && req.method === 'POST') {
            const audio = await extractAudio(req, url);
            const id = newId('ref');
            const file = refPath(id, path.extname(audio.filename || '.wav').replace(/^\./, '') || 'wav');
            if (audio.kind === 'path') fs.copyFileSync(audio.file, file);
            else fs.writeFileSync(file, audio.buffer);
            return json(res, 200, { id, path: file, filename: path.basename(file), bytes: statSafe(file).size });
        }

        if (route.startsWith('/media/') && req.method === 'GET') {
            const rest = decodeURIComponent(route.slice('/media/'.length));
            if (rest.startsWith('sample/')) {
                const dir = samplesDir(cfg);
                const name = path.basename(rest.slice('sample/'.length));
                if (!dir) return json(res, 404, { error: '未找到辨伪部署包' });
                return sendFile(res, path.join(dir, name), {
                    download: url.searchParams.get('download') === '1',
                });
            }
            if (rest.startsWith('voice/')) {
                const dir = cfg.modelDirs.gptsovits ? path.join(cfg.modelDirs.gptsovits, 'voices') : null;
                if (!dir) return json(res, 404, { error: '未找到合成部署包' });
                const name = path.basename(rest.slice('voice/'.length));
                return sendFile(res, path.join(dir, name), {
                    download: url.searchParams.get('download') === '1',
                });
            }
            const id = path.basename(rest).replace(/\.[^.]+$/, '');
            const file = mediaPath(id);
            return sendFile(res, file, { download: url.searchParams.get('download') === '1' });
        }

        // 静态前端
        const rel = route === '/' ? 'index.html' : route.replace(/^\/+/, '');
        const target = path.join(WEB_DIR, rel);
        if (!target.startsWith(WEB_DIR)) return json(res, 403, { error: '非法路径' });
        if (!statSafe(target)) return json(res, 404, { error: '页面不存在' });
        const ext = path.extname(target);
        const mime =
            ext === '.html'
                ? 'text/html; charset=utf-8'
                : ext === '.js'
                  ? 'text/javascript; charset=utf-8'
                  : ext === '.css'
                    ? 'text/css; charset=utf-8'
                    : 'application/octet-stream';
        res.writeHead(200, { 'Content-Type': mime, 'Cache-Control': 'no-store' });
        fs.createReadStream(target).pipe(res);
        return undefined;
    } catch (err) {
        console.error(`[${req.method} ${route}]`, err);
        return json(res, 500, { error: err?.message || String(err) });
    }
});

server.listen(cfg.port, cfg.host, () => {
    const lines = [
        '',
        '  语音攻防演示平台 voice-arena',
        `  页面地址   http://${cfg.host}:${cfg.port}`,
        `  合成服务   ${cfg.tts.baseUrl}${cfg.tts.path}${cfg.modelDirs.gptsovits ? '' : '（未探测到部署包）'}`,
        `  辨伪服务   ${cfg.detect.baseUrl}${cfg.detect.path}（阈值 ${cfg.detect.threshold}）`,
        `  模型目录   ${cfg.modelDirs.antispoof || '未探测到'}`,
        '',
    ];
    console.log(lines.join('\n'));
});
