/**
 * 不引入第三方库的最小 multipart 实现。
 * 上游 AASIST 服务用 FastAPI 的 UploadFile，要求标准 multipart/form-data，
 * 浏览器上传给本站的是裸字节流，这里负责把它包装成上游要的格式。
 */

/** 把一段音频包装成 multipart/form-data 请求体 */
export function buildMultipart(fieldName, filename, contentType, buffer) {
    const boundary = `----voicearena${Date.now().toString(16)}${Math.random().toString(16).slice(2, 10)}`;
    const head =
        `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="${fieldName}"; filename="${filename}"\r\n` +
        `Content-Type: ${contentType}\r\n\r\n`;
    const tail = `\r\n--${boundary}--\r\n`;
    return {
        boundary,
        body: Buffer.concat([Buffer.from(head, 'utf8'), buffer, Buffer.from(tail, 'utf8')]),
    };
}

/**
 * 解析 multipart/form-data。只取普通字段与第一个文件字段，
 * 够用即可，不追求 RFC 全覆盖。
 */
export function parseMultipart(buffer, contentType) {
    const match = /boundary=(?:"([^"]+)"|([^;]+))/i.exec(contentType || '');
    if (!match) return { fields: {}, files: {} };
    const boundary = `--${(match[1] || match[2]).trim()}`;
    const parts = splitBuffer(buffer, Buffer.from(boundary, 'utf8'));
    const fields = {};
    const files = {};
    for (const part of parts) {
        if (!part.length) continue;
        const sep = part.indexOf('\r\n\r\n');
        if (sep < 0) continue;
        const header = part.subarray(0, sep).toString('utf8');
        let body = part.subarray(sep + 4);
        // 去掉结尾的 CRLF
        if (body.length >= 2 && body[body.length - 2] === 0x0d && body[body.length - 1] === 0x0a) {
            body = body.subarray(0, body.length - 2);
        }
        const nameMatch = /name="([^"]*)"/i.exec(header);
        if (!nameMatch) continue;
        const name = nameMatch[1];
        const fileMatch = /filename="([^"]*)"/i.exec(header);
        if (fileMatch) {
            const typeMatch = /Content-Type:\s*([^\r\n]+)/i.exec(header);
            files[name] = {
                filename: fileMatch[1] || 'upload.wav',
                contentType: (typeMatch?.[1] || 'application/octet-stream').trim(),
                data: body,
            };
        } else {
            fields[name] = body.toString('utf8');
        }
    }
    return { fields, files };
}

function splitBuffer(buffer, separator) {
    const out = [];
    let start = 0;
    for (;;) {
        const idx = buffer.indexOf(separator, start);
        if (idx < 0) break;
        out.push(buffer.subarray(start, idx));
        start = idx + separator.length;
    }
    return out.map((chunk) => (chunk.length && chunk[0] === 0x0d && chunk[1] === 0x0a ? chunk.subarray(2) : chunk));
}
