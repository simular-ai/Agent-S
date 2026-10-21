// OpenCode plugin for prompt-emulated (Hermes) tool calling.
//
// When a reply mixes a parseable tool call with a malformed one, the
// tool-calling proxy keeps the malformed block as visible assistant text.
// Replayed as history, that text is imitated: measured on the s_009 opening,
// a single leaked block raised the unparseable-reply rate from 0/30 to 28/30,
// which exhausts the proxy's retries and ends the episode with HTTP 422.
//
// This hook strips such blocks from the copy of the history sent to the model
// (normal turns and compaction summaries). Stored session parts, and so the
// recorded trajectory, keep the original text.
const BLOCK = /<tool_call>[\s\S]*?(?:<\/tool_call>|$)/g;
const XML = /<function=[\s\S]*?(?:<\/function>|$)/g;

export const ShadowHermesSanitizer = async () => ({
  "experimental.chat.messages.transform": async (_input, output) => {
    let cleaned = 0;
    for (const message of output.messages ?? []) {
      if (message?.info?.role !== "assistant") continue;
      for (const part of message.parts ?? []) {
        if (part?.type !== "text" || typeof part.text !== "string") continue;
        if (!part.text.includes("<tool_call") && !part.text.includes("<function=")) continue;
        part.text = part.text.replace(BLOCK, "").replace(XML, "").trim();
        cleaned += 1;
      }
    }
    if (cleaned) console.error(`[shadow-hermes-sanitizer] cleaned ${cleaned} text part(s)`);
  },
});
