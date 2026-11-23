/// Adapts the pinned DeepSeek Harness's Cordis events to the app's line-oriented protocol.
public enum DeepSeekStreamingPlugin {
    public static let filename = "agent-s-stream.mjs"

    public static let source = #"""
    import { randomUUID } from 'node:crypto'
    import { createInterface } from 'node:readline'

    export const name = 'agent-s-live-text'

    export function apply(ctx) {
      let rootAgent
      const write = event => process.stdout.write(JSON.stringify(event) + '\n')
      const announceSteering = process.env.AGENT_S_STEERING === '1' ? setupSteering(ctx, write) : null

      ctx.on('agent/assistant-stream', ({ agent, frame }) => {
        if (!rootAgent) {
          if (frame.type !== 'start' || agent.session.header.origin === 'subagent') return
          rootAgent = agent
        }
        if (agent !== rootAgent) return

        const id = frame.attemptId
        if (frame.type === 'start') {
          write({ type: 'agent_s_text_stream', id })
          announceSteering?.(agent, frame.turn)
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

    function setupSteering(ctx, write) {
      let rootAgent
      let activeTurn
      let announced = false
      let finished = false
      let disconnected = false
      let disposed = false
      const pending = new Map()
      const result = (id, accepted) => write({ type: 'agent_s_steer_result', id, accepted })
      const stdin = process.stdin
      const input = createInterface({ input: stdin, crlfDelay: Infinity })
      const appReady = ctx.get('appReady')
      const exit = ctx.get('appExit')
      let cancelReady
      let shutdownTimer
      const disconnect = () => {
        if (disposed || disconnected || finished) return
        disconnected = true
        cancelReady = appReady.onReady(() => {
          if (disposed || finished) return
          if (rootAgent?.status !== 'running') {
            exit(130)
            return
          }
          // Give the headless runner time to flush the cancelled turn before forcing teardown.
          shutdownTimer = setTimeout(() => exit(130), 1_000).unref()
          try {
            rootAgent.cancel({ kind: 'user' })
          } catch {
            exit(130)
          }
        })
      }
      stdin.once('end', disconnect)
      stdin.once('close', disconnect)
      if (stdin.readableEnded || stdin.destroyed) queueMicrotask(disconnect)
      ctx.effect(() => () => {
        disposed = true
        cancelReady?.()
        clearTimeout(shutdownTimer)
        stdin.off('end', disconnect)
        stdin.off('close', disconnect)
        input.close()
      })

      // Identify the headless root before it sends its first model request.
      ctx.on('agent/status', ({ agent, status }) => {
        if (!rootAgent && status === 'running' && agent.session.header.origin !== 'subagent') rootAgent = agent
      })

      input.on('line', line => {
        let request
        try { request = JSON.parse(line) } catch { return }
        if (request?.type !== 'agent_s_steer' || typeof request.id !== 'string'
          || !request.id || request.id.length > 128) return
        const { id, text } = request
        const agent = activeTurn?.agent
        if (disconnected || !agent || agent.status !== 'running' || typeof text !== 'string'
          || !text.trim() || Buffer.byteLength(text, 'utf8') > 65_536) {
          result(id, false)
          return
        }
        const message = { id: randomUUID(), role: 'user', source: { kind: 'user' }, content: [{ type: 'text', text }] }
        pending.set(message.id, id)
        try {
          // Inject into the next step of the running turn without waking a new turn.
          agent.inject(message)
        } catch {
          pending.delete(message.id)
          result(id, false)
        }
      })

      ctx.on('session/event', (session, event) => {
        if (session !== rootAgent?.session) return
        if (event.type === 'turn/end') finished = true
        if (!activeTurn) return
        if (event.type === 'user/message') {
          const id = pending.get(event.data.id)
          if (pending.delete(event.data.id)) result(id, true)
        } else if (event.type === 'turn/end' && event.data.turn === activeTurn.number) {
          const { agent } = activeTurn
          activeTurn = undefined
          const uncommitted = [...pending]
          pending.clear()
          for (const [, id] of uncommitted) result(id, false)
          if (uncommitted.length) {
            // The Session cannot append while publishing turn/end. Remove any leftovers after it returns.
            queueMicrotask(() => {
              for (const [messageId] of uncommitted) agent.inbox.remove(messageId)
            })
          }
        }
      })

      return (agent, number) => {
        if (announced || disconnected || finished) return
        announced = true
        activeTurn = { agent, number }
        write({ type: 'agent_s_steer_ready' })
      }
    }
    """#
}
