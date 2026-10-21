// Exercise the pinned runtime with an in-memory provider. No HTTP or daemon.
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";

const base = pathToFileURL(`${process.argv[2]}/node_modules/`);
const load = (name) => import(new URL(name, base));
const settings = JSON.parse(process.argv[3]);
const { AgentSession } = await load("prime-agent/dist/core/agent-session.js");
const { SettingsManager } = await load("prime-agent/dist/core/settings-manager.js");
const { buildSessionContext } = await load("prime-agent/dist/core/session-manager.js");
const { calculateContextTokens, estimateContextTokens, shouldCompact, prepareCompaction, compact } =
    await load("prime-agent/dist/core/compaction/compaction.js");
const { registerApiProvider, createAssistantMessageEventStream, isContextOverflow } =
    await load("@earendil-works/pi-ai/dist/index.js");

const window = 262144;
const usage = (input, output = 0) => ({input, output, cacheRead: 0, cacheWrite: 0, totalTokens: input + output});
const assistant = (extra = {}) => ({
    role: "assistant", content: [{type: "text", text: "done"}],
    provider: "offline", model: "test", usage: usage(226000), stopReason: "stop",
    timestamp: 1000, ...extra,
});
assert.equal(calculateContextTokens({...usage(100, 20), cacheRead: 30, cacheWrite: 40, totalTokens: 0}), 190);
assert.equal(calculateContextTokens({...usage(100, 20), totalTokens: 160}), 160);
const successful = assistant();
const trailing = {role: "toolResult", toolCallId: "call", toolName: "ipython", content: [{type: "text", text: "x".repeat(200)}]};
const estimate = estimateContextTokens([successful, trailing]);
assert.equal(estimate.tokens, 226050);
assert.equal(shouldCompact(estimate.tokens, window, {...settings, reserveTokens: 16384}), false);
assert.equal(shouldCompact(estimate.tokens, window, settings), true);
assert.equal(shouldCompact(window - settings.reserveTokens, window, settings), false);
assert.equal(shouldCompact(window - settings.reserveTokens + 1, window, settings), true);

let summaryBudget;
const fakeStream = (_model, _context, options) => {
    summaryBudget = options.maxTokens;
    const stream = createAssistantMessageEventStream();
    stream.push({type: "done", reason: "stop", message: assistant({
        content: [{type: "text", text: "Offline summary retaining the task goal."}], usage: usage(100, 10),
    })});
    return stream;
};
registerApiProvider({api: "offline-test", stream: fakeStream, streamSimple: fakeStream}, "offline-test");
const model = {id: "test", provider: "offline", api: "offline-test", reasoning: false, contextWindow: window, maxTokens: 32000};
const entries = [];
for (let i = 0; i < 30; i++) {
    for (const message of [{role: "user", content: "x".repeat(12000), timestamp: i}, assistant({timestamp: i})]) {
        entries.push({type: "message", id: `entry-${entries.length}`, parentId: entries.at(-1)?.id ?? null, timestamp: new Date(i).toISOString(), message});
    }
}
const preparation = prepareCompaction(entries, settings);
assert.ok(preparation.messagesToSummarize.length > 0);
assert.equal(preparation.isSplitTurn, false);
const result = await compact(preparation, model, "offline-key");
assert.match(result.summary, /Offline summary/);
assert.equal(summaryBudget, Math.floor(0.8 * settings.reserveTokens));
const rebuilt = buildSessionContext([...entries, {
    type: "compaction", id: "compacted", parentId: entries.at(-1).id,
    timestamp: new Date(2000).toISOString(), ...result,
}]);
assert.ok(rebuilt.messages.length < entries.length);

function session(messages) {
    const dispatches = [];
    return {
        settingsManager: SettingsManager.inMemory({compaction: settings}),
        model, sessionManager: {getBranch: () => []}, agent: {state: {messages}},
        _overflowRecovery: "idle", _getThresholdContextTokens: AgentSession.prototype._getThresholdContextTokens,
        _runAutoCompaction: async (reason, retry) => {dispatches.push({reason, retry}); return true;},
        _endCompactionUnsuccessfully: () => {}, dispatches,
    };
}
const thresholdSession = session([successful, trailing]);
assert.equal(await AgentSession.prototype._checkCompaction.call(thresholdSession, successful, true, false), true);
assert.deepEqual(thresholdSession.dispatches, [{reason: "threshold", retry: false}]);

const hidden = assistant({stopReason: "error", usage: usage(0), errorMessage: "500 litellm.APIConnectionError: Hermes-toolsException - Client error '400 Bad Request'"});
const overflow = {...hidden, errorMessage: "400 litellm.BadRequestError: Hermes-toolsException - This model's maximum context length is 262144 tokens. However, you requested 267282 tokens."};
assert.equal(isContextOverflow(hidden, window), false);
assert.equal(isContextOverflow(overflow, window), true);
assert.equal(await AgentSession.prototype._checkCompaction.call(session([hidden]), hidden, true, false), false);
const recoverySession = session([overflow]);
assert.equal(await AgentSession.prototype._checkCompaction.call(recoverySession, overflow, true, false), true);
assert.deepEqual(recoverySession.dispatches, [{reason: "overflow", retry: true}]);
assert.deepEqual(recoverySession.agent.state.messages, []); // Error is kept in session history, removed from retry input.
assert.equal(await AgentSession.prototype._checkCompaction.call(recoverySession, overflow, true, false), false);
assert.equal(recoverySession.dispatches.length, 1); // One compact-and-retry attempt, not an infinite loop.
console.log(JSON.stringify({
    thresholdTokens: estimate.tokens, summaryBudget,
    compactedMessageCount: rebuilt.messages.length, originalMessageCount: entries.length,
    overflowDispatch: recoverySession.dispatches[0],
}));
