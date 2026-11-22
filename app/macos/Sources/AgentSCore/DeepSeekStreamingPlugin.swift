/// Adapts the pinned DeepSeek Harness's Cordis events to the app's line-oriented stream.
public enum DeepSeekStreamingPlugin {
    public static let filename = "agent-s-stream.mjs"

    public static let source = #"""
    export const name = 'agent-s-live-text'

    export function apply(ctx) {
      let rootAgent
      const write = event => process.stdout.write(JSON.stringify(event) + '\n')

      ctx.on('agent/assistant-stream', ({ agent, frame }) => {
        if (!rootAgent) {
          if (frame.type !== 'start' || agent.session.header.origin === 'subagent') return
          rootAgent = agent
        }
        if (agent !== rootAgent) return

        const id = frame.attemptId
        if (frame.type === 'start') {
          write({ type: 'agent_s_text_stream', id })
        } else if (frame.type === 'chunk' && frame.chunk.type === 'text-delta' && frame.chunk.text !== '') {
          write({ type: 'agent_s_text_delta', id, text: frame.chunk.text })
        } else if (frame.type === 'end') {
          let text = ''
          if (frame.outcome.kind === 'committed' && frame.outcome.eventType === 'assistant/message') {
            const event = agent.session.eventAt(frame.outcome.seq)
            if (event?.type === 'assistant/message') {
              text = event.data.message.content
                .filter(block => block.type === 'text')
                .map(block => block.text)
                .join('')
            }
          }
          // The committed text replaces the provisional reply. Remove any text from a failed attempt.
          write({ type: 'agent_s_text_snapshot', id, text })
        }
      })
    }
    """#
}
