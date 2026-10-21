// OpenCode plugin: cap the bash tool's model-chosen `timeout`.
//
// The pinned CLI honours any timeout the model passes and emits nothing while
// a tool runs. A GUI launched with its stdout still attached then holds the
// call open for that long, and the supervisor's inactivity watchdog ends the
// whole episode (if-002 s_007 and 014: idle_timeout). With a cap, the CLI
// terminates the command at the cap and the agent continues.
export const ShadowToolLimits = async (_input, options = {}) => {
  const cap = Number(options.maxBashTimeoutMs);
  if (!Number.isFinite(cap) || cap <= 0) return {};
  return {
    "tool.execute.before": async (input, output) => {
      if (input?.tool !== "bash" || !output?.args) return;
      const requested = Number(output.args.timeout);
      if (Number.isFinite(requested) && requested > cap) {
        output.args.timeout = cap;
        console.error(`[shadow-tool-limits] bash timeout ${requested}ms capped to ${cap}ms`);
      }
    },
  };
};
