// Projects the pinned DeepSeek SDK's root assistant stream into Agent S's JSON-RPC connection.
export const name = 'agent-s-live-text'

export function apply(ctx) {
  let root
  const write = params => process.stdout.write(JSON.stringify({
    jsonrpc: '2.0', method: 'agent_s/stream', params,
  }) + '\n')
  write({ type: 'ready' })

  ctx.on('agent/assistant-stream', ({ agent, frame }) => {
    if (!root && frame.type === 'start' && agent.session.header.origin !== 'subagent') root = agent
    if (agent !== root) return

    const id = frame.attemptId
    if (frame.type === 'chunk' && frame.chunk.type === 'text-delta' && frame.chunk.text) {
      write({ type: 'delta', id, text: frame.chunk.text })
    } else if (frame.type === 'end') {
      // The committed message replaces streamed text, including when an attempt was discarded.
      const event = frame.outcome.kind === 'committed' && frame.outcome.eventType === 'assistant/message'
        ? agent.session.eventAt(frame.outcome.seq) : null
      const text = event?.type === 'assistant/message'
        ? event.data.message.content.filter(block => block.type === 'text').map(block => block.text).join('')
        : ''
      write({ type: 'snapshot', id, text })
    }
  })
}
