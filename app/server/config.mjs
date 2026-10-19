/**
 * 运行配置。默认值写在代码里，config.json 存在时覆盖默认值。
 * 模型目录按 E:\voice_ws 下的 *_deploy_* 目录自动探测，避免手工改路径。
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));

export const ROOT = path.resolve(HERE, '..');
export const DATA_DIR = path.join(ROOT, 'data');
export const MEDIA_DIR = path.join(DATA_DIR, 'media');
export const REF_DIR = path.join(DATA_DIR, 'refs');
export const WEB_DIR = path.join(ROOT, 'web');
export const RUNS_FILE = path.join(DATA_DIR, 'runs.json');
export const CONFIG_FILE = path.join(ROOT, 'config.json');
export const WORKSPACE = path.resolve(ROOT, '..');

/** 在 E:\voice_ws 下找形如 xxx_deploy_yyyymmdd 的目录 */
function detectDeployDir(keyword) {
    try {
        const hit = fs
            .readdirSync(WORKSPACE, { withFileTypes: true })
            .filter((e) => e.isDirectory() && e.name.toLowerCase().includes(keyword))
            .map((e) => path.join(WORKSPACE, e.name))
            .sort((a, b) => b.length - a.length)[0];
        return hit ?? null;
    } catch {
        return null;
    }
}

const defaults = {
    host: '127.0.0.1',
    port: 8790,
    tts: {
        // GPT-SoVITS 推理服务，默认端口 9880
        baseUrl: 'http://127.0.0.1:9880',
        path: '/tts',
        timeoutMs: 180000,
        defaults: {
            text_lang: 'zh',
            prompt_lang: 'zh',
            prompt_text: '',
            ref_audio_path: null,
            media_type: 'wav',
            text_split_method: 'cut5',
            batch_size: 1,
            speed_factor: 1.0,
            temperature: 1.0,
            top_k: 15,
            top_p: 1.0,
            repetition_penalty: 1.35,
            streaming_mode: false,
        },
    },
    detect: {
        // AASIST 辨伪服务，默认端口 9890
        baseUrl: 'http://127.0.0.1:9890',
        path: '/detect',
        healthPath: '/health',
        timeoutMs: 120000,
        threshold: -1.09385,
    },
    modelDirs: {
        antispoof: detectDeployDir('antispoof'),
        gptsovits: detectDeployDir('gptsovits'),
    },
    access: {
        // 令牌留空则不做限制；填了之后，非本机 Host 的请求必须带上它。
        // 这样本地调试不用带参数，公网入口默认是关着的。
        token: '',
    },
};

function deepMerge(base, extra) {
    if (!extra || typeof extra !== 'object' || Array.isArray(extra)) return base;
    const out = Array.isArray(base) ? [...base] : { ...base };
    for (const [k, v] of Object.entries(extra)) {
        out[k] = v && typeof v === 'object' && !Array.isArray(v) ? deepMerge(base?.[k] ?? {}, v) : v;
    }
    return out;
}

export function loadConfig() {
    let fromFile = null;
    if (fs.existsSync(CONFIG_FILE)) {
        try {
            fromFile = JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf8'));
        } catch (err) {
            console.warn('[config] config.json 解析失败，改用默认配置:', err.message);
        }
    }
    const cfg = deepMerge(defaults, fromFile);
    // 模型目录没被显式覆盖时，再探测一次，避免首次启动时目录还没解压
    cfg.modelDirs.antispoof ||= detectDeployDir('antispoof');
    cfg.modelDirs.gptsovits ||= detectDeployDir('gptsovits');
    return cfg;
}

export function ensureDirs() {
    for (const dir of [DATA_DIR, MEDIA_DIR, REF_DIR]) {
        fs.mkdirSync(dir, { recursive: true });
    }
}

/** 辨伪包里自带的样例音频目录 */
export function samplesDir(cfg) {
    const base = cfg.modelDirs.antispoof;
    return base ? path.join(base, 'samples') : null;
}
