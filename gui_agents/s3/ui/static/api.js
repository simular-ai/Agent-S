/** Typed DOM + fetch helpers shared by all UI panels. */
/** Get an element by id, cast to the expected HTMLElement subtype. */
export function el(id) {
    const node = document.getElementById(id);
    if (!node)
        throw new Error(`missing element #${id}`);
    return node;
}
export function inputVal(id) {
    return el(id).value;
}
export function inputNum(id, fallback) {
    const v = parseFloat(el(id).value);
    return Number.isFinite(v) ? v : fallback;
}
export function checkVal(id) {
    return el(id).checked;
}
/** Inline status line under a panel (ok / err / warn styling). */
export function msg(id, text, cls = "") {
    const target = el(id);
    target.textContent = text;
    target.className = `msg ${cls}`;
}
/** JSON fetch that throws an Error with the server's `detail` on failure. */
export async function api(path, opts = {}) {
    const token = document.querySelector('meta[name="agent-s-token"]')?.content;
    if (!token)
        throw new Error("Missing UI authentication token; reload the page");
    const headers = new Headers(opts.headers);
    headers.set("Authorization", `Bearer ${token}`);
    headers.set("Content-Type", "application/json");
    const res = await fetch(path, {
        ...opts,
        headers,
        credentials: "same-origin",
    });
    const text = await res.text();
    let data;
    try {
        data = JSON.parse(text);
    }
    catch {
        data = { raw: text };
    }
    if (!res.ok) {
        const detail = typeof data === "object" && data !== null && "detail" in data
            ? String(data.detail)
            : text.slice(0, 300);
        throw new Error(detail);
    }
    return data;
}
export async function screenshot(taskId, signal) {
    const token = document.querySelector('meta[name="agent-s-token"]')?.content;
    if (!token)
        throw new Error("Missing authentication token");
    const response = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/screenshot`, { signal, headers: { Authorization: `Bearer ${token}` }, cache: "no-store" });
    if (!response.ok)
        throw new Error(`Screenshot request failed (${response.status})`);
    return response.blob();
}
