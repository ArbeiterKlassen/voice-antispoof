#!/usr/bin/env node
/**
 * Cloudflare Tunnel 启动器（由 start-tunnel.bat 调用，bat 只放 ASCII，中文都从这里输出）。
 *
 * 做四件事：
 *   1. 找到 cloudflared（CLOUDFLARED 环境变量 → PATH → 常见安装位置）
 *   2. 读隧道配置，取出源地址与公网主机名
 *   3. 预检本机站点是否在跑，没跑就提示先开 start-voice-arena.bat
 *   4. 前台运行 cloudflared，输出同时写进 logs/tunnel.out.log（Ctrl+C 停）
 */
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync, spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const LOG_DIR = path.join(ROOT, 'logs');
const TUNNEL_LOG = path.join(LOG_DIR, 'tunnel.out.log');
const CONFIG = path.join(ROOT, 'data', 'cloudflared', 'config.yml');

const argv = process.argv.slice(2);
const has = (name) => argv.includes(`--${name}`);
const NO_AUTOSTART = has('no-autostart');

const c = {
    cyan: (t) => `\u001b[36m${t}\u001b[0m`,
    green: (t) => `\u001b[32m${t}\u001b[0m`,
    yellow: (t) => `\u001b[33m${t}\u001b[0m`,
    red: (t) => `\u001b[31m${t}\u001b[0m`,
    dim: (t) => `\u001b[2m${t}\u001b[0m`,
};

console.log('');
console.log(c.cyan('  =========================================='));
console.log('    voice-arena · Cloudflare Tunnel');
console.log(c.cyan('  =========================================='));
console.log('');

/* 1. 找 cloudflared */
function findCloudflared() {
    const candidates = [
        process.env.CLOUDFLARED,
        'cloudflared',
        'D:\\multi-modal-ai\\tools\\cloudflared\\cloudflared.exe',
        'C:\\Program Files (x86)\\cloudflared\\cloudflared.exe',
        'C:\\Program Files\\cloudflared\\cloudflared.exe',
    ].filter(Boolean);
    for (const bin of candidates) {
        if (bin !== 'cloudflared' && !fs.existsSync(bin)) continue;
        try {
            execFileSync(bin, ['--version'], { stdio: 'pipe', windowsHide: true, timeout: 10000 });
            return bin;
        } catch {
            /* 试下一个候选 */
        }
    }
    return null;
}

const CF = findCloudflared();
if (!CF) {
    console.error(c.red('  [错误] 找不到 cloudflared。'));
    console.error(c.dim('         装一个：winget install Cloudflare.cloudflared'));
    console.error(c.dim('         或下载 exe 后设置环境变量 CLOUDFLARED=<路径>'));
    process.exit(1);
}

/* 2. 读配置 */
if (!fs.existsSync(CONFIG)) {
    console.error(c.red(`  [错误] 找不到隧道配置：${CONFIG}`));
    console.error(c.dim('         可从 docs/cloudflare-tunnel.example.yml 复制一份改好。'));
    process.exit(1);
}
const configText = fs.readFileSync(CONFIG, 'utf8');
const serviceLine = configText.split(/\r?\n/).find((l) => /^\s*service:\s*\S+/.test(l)) ?? '';
const serviceUrl = (serviceLine.split('service:')[1] ?? '').trim();
const hosts = [...configText.matchAll(/^\s*-\s*hostname:\s*(\S+)/gm)].map((m) => m[1]);
const publicHost = hosts[0] ?? '';

// 站点端口从 config.json 读，避免这里和站点配置各写一份
let sitePort = 8790;
try {
    const siteCfg = JSON.parse(fs.readFileSync(path.join(ROOT, 'config.json'), 'utf8'));
    sitePort = siteCfg.port ?? sitePort;
} catch {
    /* 用默认值 */
}

console.log(`  隧道配置   ${path.relative(ROOT, CONFIG)}`);
console.log(`  源地址     ${c.cyan(serviceUrl)}`);
console.log(`  公网入口   ${c.green(`https://${publicHost}`)}`);
console.log('');

/* 3. 预检本机站点 */
async function probe(url, timeoutMs = 5000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        // 任何 HTTP 响应都说明服务在监听；带令牌门时 401 也算活着
        const res = await fetch(url, { signal: controller.signal });
        return res.status < 500;
    } catch {
        return false;
    } finally {
        clearTimeout(timer);
    }
}

const probeUrl = `http://127.0.0.1:${sitePort}/api/state`;
if (await probe(probeUrl)) {
    console.log(c.green(`  [就绪] 本机站点在线：${probeUrl}`));
} else {
    console.log(c.yellow(`  [警告] 本机站点没响应：${probeUrl}`));
    if (NO_AUTOSTART) {
        console.log(c.dim('         加 --no-autostart 时不会自动拉服务，请自行启动。'));
    } else {
        console.log(c.dim('         请先运行 start-voice-arena.bat（另一个窗口）。'));
    }
    console.log(c.dim('         隧道仍会启动，站点起来后公网即可访问。'));
}
console.log('');

/* 4. 前台运行 */
fs.mkdirSync(LOG_DIR, { recursive: true });
console.log(c.dim(`  日志       ${path.relative(ROOT, TUNNEL_LOG)}`));
console.log(c.dim('  按 Ctrl+C 停止隧道'));
console.log('');

const child = spawn(CF, ['tunnel', '--config', CONFIG, 'run'], {
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
});
const log = fs.createWriteStream(TUNNEL_LOG, { flags: 'a' });
const relay = (stream, sink) => {
    stream.on('data', (chunk) => {
        sink.write(chunk);
        log.write(chunk);
    });
};
relay(child.stdout, process.stdout);
relay(child.stderr, process.stderr);

const shutdown = () => {
    log.end();
    child.kill();
};
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
child.on('exit', (code) => {
    log.end();
    console.log('');
    console.log(c.yellow(`  隧道退出，code=${code}`));
    process.exit(code ?? 0);
});
