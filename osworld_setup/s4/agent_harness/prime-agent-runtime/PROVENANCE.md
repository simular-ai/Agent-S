# Prime Agent Python runtime provenance

This directory vendors the Python runtime from
[`PrimeIntellect-ai/prime-agent`](https://github.com/PrimeIntellect-ai/prime-agent)
at commit `a9b5d88b50db50860088772a408e867f8071b507`.

Copied paths:

- repository `LICENSE`
- `prime-agent-runtime/pyproject.toml`
- `prime-agent-runtime/uv.lock`
- `prime-agent-runtime/src/rlm/*.py`

The files are copied without source changes. `SOURCE_MANIFEST.json` records the
SHA-256 digest of every copied file, and
`agent_s4.harness.prime.source.validate_prime_agent_runtime_source`
checks the pinned commit, required file set, digests, and unexpected importable
artifacts before this runtime is used.

## Auto-compaction boundary

Prime Agent's automatic context compaction is not implemented by this Python
package. It is owned by the upstream TypeScript CLI in
`packages/coding-agent/src/core/agent-session.ts`, with summary construction in
`packages/coding-agent/src/core/compaction/compaction.ts` and defaults in
`packages/coding-agent/src/core/settings-manager.ts`. The integration was
audited against these upstream files at the pinned commit (not vendored):

| Upstream path | SHA-256 |
|---|---|
| `packages/ai/src/utils/overflow.ts` | `0f3abf66b9d1112523fe793976a6ccfc2f94553bd0fc8a59fa0038e0ea1cfee1` |
| `packages/coding-agent/src/core/agent-session.ts` | `d0142a96a0a3131aa24baa7f79792b07ac4f6e5bafd793570ed9569d87e47e32` |
| `packages/coding-agent/src/core/compaction/branch-summarization.ts` | `ecedee48d4bad76e0a3c1343b960d94cd27d0d431b4c4ef8fb172c9f4754fcc0` |
| `packages/coding-agent/src/core/compaction/compaction.ts` | `7df1be327e779ea9f29184848d96a6575f81196e3e3fb8f3c14dc5d6ed54e929` |
| `packages/coding-agent/src/core/compaction/index.ts` | `c84724a0a7482afa25e1fea50c93a6d4f402f609f5527d86cc8f80f07d0b48c5` |
| `packages/coding-agent/src/core/compaction/utils.ts` | `37f025ad29ea81fb3d07ac37a8103ee8bc1f65c765c8263d68a96ae1e39f85e5` |
| `packages/coding-agent/src/core/settings-manager.ts` | `29f25f09d7975221621137b96fa7cbd05b1ad6007fee677e7d64e70fc687fc86` |

Consequently, using the upstream Prime Agent CLI artifact built at the pinned revision retains
threshold compaction, overflow compaction-and-retry, and the persistent IPython
kernel across compaction. Importing or installing only this vendored Python
runtime does not provide auto-compaction by itself. An integration must launch
the upstream CLI/session host and must not set `compaction.enabled` to `false`.
