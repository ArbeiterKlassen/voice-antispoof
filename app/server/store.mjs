/**
 * 轻量存储：运行记录写成一个 JSON 文件，音频文件落地在 data/media。
 * 演示场景数据量很小，用文件足够，也不引入数据库依赖。
 */
import fs from 'node:fs';
import path from 'node:path';
import { MEDIA_DIR, REF_DIR, RUNS_FILE } from './config.mjs';

let runs = null;

function readRuns() {
    if (runs) return runs;
    try {
        runs = JSON.parse(fs.readFileSync(RUNS_FILE, 'utf8'));
        if (!Array.isArray(runs)) runs = [];
    } catch {
        runs = [];
    }
    return runs;
}

function persist() {
    fs.writeFileSync(RUNS_FILE, JSON.stringify(runs, null, 2), 'utf8');
}

export function listRuns() {
    return readRuns();
}

export function addRun(record) {
    const all = readRuns();
    all.unshift(record);
    // 只保留最近 200 条，避免演示久了文件无限增长
    if (all.length > 200) all.length = 200;
    persist();
    return record;
}

export function clearRuns() {
    runs = [];
    persist();
}

let counter = 0;

/** 生成一个按时间排序、便于人读的文件 id */
export function newId(prefix = 'a') {
    counter = (counter + 1) % 1000;
    const d = new Date();
    const pad = (n, w = 2) => String(n).padStart(w, '0');
    const stamp = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(
        d.getMinutes(),
    )}${pad(d.getSeconds())}`;
    return `${prefix}_${stamp}_${pad(counter, 3)}`;
}

export function mediaPath(id, ext = 'wav') {
    return path.join(MEDIA_DIR, `${id}.${ext}`);
}

export function refPath(id, ext = 'wav') {
    return path.join(REF_DIR, `${id}.${ext}`);
}

export function statSafe(file) {
    try {
        return fs.statSync(file);
    } catch {
        return null;
    }
}
