/** Typed DOM + fetch helpers shared by all UI panels. */

/** Get an element by id, cast to the expected HTMLElement subtype. */
export function el<T extends HTMLElement = HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node as T;
}

export function inputVal(id: string): string {
  return el<HTMLInputElement>(id).value;
}

export function inputNum(id: string, fallback: number): number {
  const v = parseFloat(el<HTMLInputElement>(id).value);
  return Number.isFinite(v) ? v : fallback;
}

export function checkVal(id: string): boolean {
  return el<HTMLInputElement>(id).checked;
}

/** Inline status line under a panel (ok / err / warn styling). */
export function msg(id: string, text: string, cls = ""): void {
  const target = el(id);
  target.textContent = text;
  target.className = `msg ${cls}`;
}

/** JSON fetch that throws an Error with the server's `detail` on failure. */
export async function api<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const token = document.querySelector<HTMLMetaElement>('meta[name="agent-s-token"]')?.content;
  if (!token) throw new Error("Missing UI authentication token; reload the page");
  const headers = new Headers(opts.headers);
  headers.set("Authorization", `Bearer ${token}`);
  headers.set("Content-Type", "application/json");
  const res = await fetch(path, {
    ...opts,
    headers,
    credentials: "same-origin",
  });
  const text = await res.text();
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    data = { raw: text };
  }
  if (!res.ok) {
    const detail =
      typeof data === "object" && data !== null && "detail" in data
        ? String((data as Record<string, unknown>).detail)
        : text.slice(0, 300);
    throw new Error(detail);
  }
  return data as T;
}

export async function screenshot(taskId: string, signal: AbortSignal): Promise<Blob> {
  const token = document.querySelector<HTMLMetaElement>('meta[name="agent-s-token"]')?.content;
  if (!token) throw new Error("Missing authentication token");
  const response = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/screenshot`, { signal, headers: { Authorization: `Bearer ${token}` }, cache: "no-store" });
  if (!response.ok) throw new Error(`Screenshot request failed (${response.status})`);
  return response.blob();
}
