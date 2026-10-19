/** 语音攻防演示平台前端：全部原生 API，无构建步骤。 */

const $ = (sel) => document.querySelector(sel);
const state = { samples: [], voices: [], prompts: {}, lastAudioId: null, threshold: -1.09385 };

/** 分数刻度的显示范围，AASIST 的 bonafide logit 基本落在这一区间 */
const SCALE_MIN = -6;
const SCALE_MAX = 6;

function toast(message, kind = 'info') {
    const el = document.createElement('div');
    el.className = `toast${kind === 'error' ? ' error' : ''}`;
    el.textContent = message;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), kind === 'error' ? 6000 : 3200);
}

async function api(path, options) {
    const res = await fetch(path, options);
    const text = await res.text();
    let data;
    try {
        data = text ? JSON.parse(text) : {};
    } catch {
        throw new Error(`服务端返回了非 JSON 内容：${text.slice(0, 200)}`);
    }
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
}

function fmtScore(v) {
    const n = Number(v);
    return Number.isFinite(n) ? (n >= 0 ? `+${n.toFixed(3)}` : n.toFixed(3)) : '—';
}

function fmtMs(ms) {
    return ms == null ? '—' : ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(2)} s`;
}

// ------------------------------------------------------------------ 状态
async function loadState() {
    try {
        const s = await api('/api/state');
        state.threshold = Number(s.detect.threshold ?? state.threshold);
        const tts = $('.pill[data-role="tts"]');
        const det = $('.pill[data-role="detect"]');
        tts.textContent = `合成 ${s.tts.up ? '在线' : '离线'} · ${s.tts.baseUrl.replace(/^https?:\/\//, '')}`;
        tts.className = `pill ${s.tts.up ? 'up' : 'down'}`;
        det.textContent = `辨伪 ${s.detect.up ? '在线' : '离线'} · 阈值 ${state.threshold.toFixed(6)}`;
        det.className = `pill ${s.detect.up ? 'up' : 'down'}`;
        $("#footInfo").textContent =
            `voice-arena 0.1.0 · 主机 ${s.app.hostname} · 辨伪设备 ${s.detect.device || '未上报'} · ` +
            `模型目录 ${s.modelDirs.antispoof || '未探测到'} · 记录 ${s.runs} 条`;
        if (!s.detect.up) toast('辨伪服务未启动，请先运行 start-services.bat', 'error');
        return s;
    } catch (err) {
        toast(`状态获取失败：${err.message}`, 'error');
        return null;
    }
}

// ------------------------------------------------------------------ 样例
async function loadSamples() {
    const box = $('#samples');
    try {
        const { items, voices = [] } = await api('/api/samples');
        state.samples = items;
        state.voices = voices;
        if (!items.length) {
            box.innerHTML = '<p class="hint">未找到样例音频，检查辨伪部署包是否解压到 E:\\voice_ws。</p>';
            return;
        }
        box.innerHTML = '';
        for (const s of items) {
            const card = document.createElement('div');
            card.className = 'sample';
            card.innerHTML = `
                <div class="head">
                    <span class="name">${s.name}</span>
                    <span class="score">实测 ${fmtScore(s.known)} · 判 ${s.expect || '?'}</span>
                </div>
                <div class="tag">${s.label || ''}${s.kind ? ` · ${s.kind}` : ''}</div>
                <div class="row">
                    <button data-act="play">播放</button>
                    <button data-act="detect">送检</button>
                    <a class="upload inline" href="/media/sample/${encodeURIComponent(s.name)}?download=1">下载</a>
                </div>
                <div class="player"></div>`;
            card.querySelector('[data-act="play"]').onclick = () => {
                const holder = card.querySelector('.player');
                holder.innerHTML = `<audio controls autoplay src="/media/sample/${encodeURIComponent(s.name)}"></audio>`;
            };
            card.querySelector('[data-act="detect"]').onclick = () => detect({ media: `sample:${s.name}` });
            box.appendChild(card);
        }
        // 参考音频下拉：合成包自带的参考音排最前，其次是对应说话人的真声样例
        state.prompts = {};
        const ref = $('#ref');
        ref.innerHTML = '';
        const addOption = (value, text, selected, promptText) => {
            const opt = document.createElement('option');
            opt.value = value;
            opt.textContent = text;
            if (selected) opt.selected = true;
            if (promptText) state.prompts[value] = promptText;
            ref.appendChild(opt);
        };
        for (const v of voices) {
            addOption(
                `ttsvoice:${v.name}`,
                `${v.name}（合成包参考音${v.promptText ? '，自带逐字文本' : ''}）`,
                v.name === 'ref_user1.wav',
                v.promptText,
            );
        }
        for (const s of items) addOption(`sample:${s.name}`, `${s.name}（${s.label || ''}）`, false, '');
        addOption('custom', '（使用上传的参考音频）', false, '');
        ref.querySelector('option[value="custom"]').dataset.custom = '1';
        applyPrompt();
    } catch (err) {
        box.innerHTML = `<p class="hint">样例列表加载失败：${err.message}</p>`;
    }
}

/** 选到自带逐字文本的参考音时，自动把 prompt 文本填上 */
function applyPrompt() {
    const text = state.prompts?.[$('#ref').value];
    if (text) $('#refText').value = text;
}

// ------------------------------------------------------------------ 结果卡片
function renderDetect(result, elapsedMs) {
    const attacked = result.attacked ?? result.verdict === 'real';
    const pct = (v) => `${((Math.max(SCALE_MIN, Math.min(SCALE_MAX, v)) - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)) * 100}%`;
    const thrPct = pct(result.threshold_aasist);
    const scorePct = pct(result.aasist_score);
    return `
        <div class="verdict ${attacked ? 'attacked' : 'blocked'}">
            <span class="big">${attacked ? '攻击成功' : '被拦截'}</span>
            <span class="why">
                判定「${result.verdict === 'real' ? '真声 real' : '伪造 fake'}」，
                分数 ${fmtScore(result.aasist_score)} ${attacked ? '≥' : '<'} 阈值 ${fmtScore(result.threshold_aasist)}，
                余量 ${fmtScore(result.margin)}${elapsedMs ? ` · 耗时 ${fmtMs(elapsedMs)}` : ''}
            </span>
        </div>
        <div class="meter">
            <div class="bar">
                <span class="thr" style="left:${thrPct}"></span>
                <span class="marker" style="left:${scorePct}"></span>
            </div>
            <div class="legend">
                <span>${SCALE_MIN}</span><span>阈值 ${fmtScore(result.threshold_aasist)}</span><span>${SCALE_MAX}</span>
            </div>
        </div>
        <dl class="kv">
            <dt>AASIST 分数</dt><dd>${fmtScore(result.aasist_score)}（判 ${result.aasist_verdict}）</dd>
            <dt>AASIST-L 分数</dt><dd>${fmtScore(result.aasist_l_score)}（本包未标定阈值，仅报分）</dd>
            <dt>口径</dt><dd>${result.note || ''}</dd>
        </dl>`;
}

// ------------------------------------------------------------------ 合成
function collectTtsParams() {
    const params = {
        speed_factor: Number($('#speed').value) || 1.0,
        temperature: Number($('#temp').value) || 1.0,
        top_k: Number($('#topk').value) || 15,
    };
    const refText = $('#refText').value.trim();
    if (refText) params.prompt_text = refText;
    // ref 交给服务端解析成上游可读的绝对路径
    return { text: $('#text').value.trim(), ref: $('#ref').value, params };
}

async function synthOnly() {
    const body = collectTtsParams();
    if (!body.text) return toast('请先填写合成文本', 'error');
    const btn = $('#btnTts');
    btn.disabled = true;
    btn.textContent = '合成中…';
    try {
        const rec = await api('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        state.lastAudioId = rec.audio.id;
        $('#btnDetectLast').disabled = false;
        $('#ttsOut').innerHTML = `
            <audio controls src="${rec.audio.url}"></audio>
            <dl class="kv">
                <dt>音频</dt><dd>${rec.audio.id} · ${(rec.audio.bytes / 1024).toFixed(1)} KB</dd>
                <dt>耗时</dt><dd>${fmtMs(rec.elapsedMs)}</dd>
                <dt>调用方式</dt><dd>${rec.via}</dd>
                <dt>下载</dt><dd><a href="${rec.audio.url}?download=1">保存 wav</a></dd>
            </dl>`;
        toast('合成完成');
        loadRuns();
    } catch (err) {
        $('#ttsOut').innerHTML = `<p class="hint">合成失败：${err.message}</p>`;
        toast(`合成失败：${err.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '仅合成';
    }
}

async function fullAttack() {
    const body = collectTtsParams();
    if (!body.text) return toast('请先填写合成文本', 'error');
    const btn = $('#btnAttack');
    btn.disabled = true;
    btn.textContent = '回合进行中…';
    try {
        const rec = await api('/api/attack', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        state.lastAudioId = rec.audio.id;
        $('#btnDetectLast').disabled = false;
        $('#ttsOut').innerHTML = `
            <audio controls src="${rec.audio.url}"></audio>
            <dl class="kv">
                <dt>克隆音频</dt><dd>${rec.audio.id} · ${(rec.audio.bytes / 1024).toFixed(1)} KB</dd>
                <dt>合成耗时</dt><dd>${fmtMs(rec.ttsMs)}（${rec.via}）</dd>
                <dt>检测耗时</dt><dd>${fmtMs(rec.detectMs)}</dd>
            </dl>`;
        $('#detOut').innerHTML = renderDetect(rec.result, rec.detectMs);
        toast(rec.outcome === 'attacked' ? '本轮攻击成功：克隆语音被判为真声' : '本轮被拦截：克隆语音被判为伪造');
        loadRuns();
    } catch (err) {
        toast(`对抗回合失败：${err.message}`, 'error');
        $('#detOut').innerHTML = `<p class="hint">回合失败：${err.message}</p>`;
    } finally {
        btn.disabled = false;
        btn.textContent = '合成并送检（完整回合）';
    }
}

// ------------------------------------------------------------------ 辨伪
async function detect({ file = null, media = null } = {}) {
    const out = $('#detOut');
    out.innerHTML = '<p class="hint">检测中…</p>';
    try {
        let rec;
        if (media) {
            rec = await api(`/api/detect?media=${encodeURIComponent(media)}`, { method: 'POST' });
        } else {
            rec = await api(`/api/detect?filename=${encodeURIComponent(file.name)}`, {
                method: 'POST',
                headers: { 'Content-Type': file.type || 'application/octet-stream' },
                body: file,
            });
        }
        out.innerHTML = renderDetect(rec.result, rec.elapsedMs);
        toast(rec.result.attacked ? '判定：真声（攻击成功）' : '判定：伪造（被拦截）');
        loadRuns();
    } catch (err) {
        out.innerHTML = `<p class="hint">检测失败：${err.message}</p>`;
        toast(`检测失败：${err.message}`, 'error');
    }
}

// ------------------------------------------------------------------ 记录
async function loadRuns() {
    const box = $('#runs');
    try {
        const { items } = await api('/api/runs');
        if (!items.length) {
            box.innerHTML = '<p class="hint">还没有记录。</p>';
            return;
        }
        box.innerHTML = '';
        for (const r of items) {
            const el = document.createElement('div');
            el.className = 'run';
            if (r.kind === 'attack') {
                el.innerHTML = `
                    <div class="row1">
                        <span class="id">${r.id}</span>
                        <span class="badge ${r.outcome}">${r.outcome === 'attacked' ? '攻击成功' : '被拦截'}</span>
                    </div>
                    <div>分数 ${fmtScore(r.result.aasist_score)} · 阈值 ${fmtScore(r.result.threshold_aasist)} · 检测 ${fmtMs(r.detectMs)}</div>
                    <div class="text">${r.text}</div>
                    <audio controls src="${r.audio.url}"></audio>`;
            } else if (r.kind === 'detect') {
                el.innerHTML = `
                    <div class="row1">
                        <span class="id">${r.id}</span>
                        <span class="badge ${r.result.attacked ? 'attacked' : 'blocked'}">${r.result.verdict}</span>
                    </div>
                    <div>来源 ${r.source || '上传'} · 分数 ${fmtScore(r.result.aasist_score)} · ${fmtMs(r.elapsedMs)}</div>`;
            } else {
                el.innerHTML = `
                    <div class="row1">
                        <span class="id">${r.id}</span>
                        <span class="badge">合成</span>
                    </div>
                    <div>${(r.audio.bytes / 1024).toFixed(1)} KB · ${fmtMs(r.elapsedMs)} · ${r.via}</div>
                    <div class="text">${r.text}</div>
                    <audio controls src="${r.audio.url}"></audio>`;
            }
            box.appendChild(el);
        }
    } catch (err) {
        box.innerHTML = `<p class="hint">记录加载失败：${err.message}</p>`;
    }
}

// ------------------------------------------------------------------ 绑定
function bind() {
    $('#refresh').onclick = () => {
        loadState();
        loadSamples();
        loadRuns();
    };
    $('#btnTts').onclick = synthOnly;
    $('#btnAttack').onclick = fullAttack;
    $('#btnDetectLast').onclick = () => {
        if (state.lastAudioId) detect({ media: `media:${state.lastAudioId}` });
    };
    $('#btnRuns').onclick = loadRuns;
    $('#ref').onchange = applyPrompt;
    $('#btnClear').onclick = async () => {
        if (!confirm('清空全部对抗记录？音频文件保留。')) return;
        await api('/api/runs', { method: 'DELETE' });
        loadRuns();
    };
    $('#detFile').onchange = (e) => {
        if (e.target.files[0]) detect({ file: e.target.files[0] });
    };
    $('#refFile').onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        try {
            const rec = await api(`/api/refs?filename=${encodeURIComponent(file.name)}`, {
                method: 'POST',
                headers: { 'Content-Type': file.type || 'application/octet-stream' },
                body: file,
            });
            const ref = $('#ref');
            const opt = ref.querySelector('option[data-custom="1"]');
            opt.value = `ref:${rec.id}`;
            opt.textContent = `上传的参考音频（${rec.filename}）`;
            opt.selected = true;
            toast('参考音频已上传');
        } catch (err) {
            toast(`参考音频上传失败：${err.message}`, 'error');
        }
    };
    const drop = $('#drop');
    drop.ondragover = (e) => {
        e.preventDefault();
        drop.classList.add('over');
    };
    drop.ondragleave = () => drop.classList.remove('over');
    drop.ondrop = (e) => {
        e.preventDefault();
        drop.classList.remove('over');
        const file = e.dataTransfer?.files?.[0];
        if (file) detect({ file });
    };
}

bind();
loadState();
loadSamples();
loadRuns();
setInterval(loadState, 15000);
