// The agent's screen verbs that need the accessibility tree, run by /usr/local/bin/ui.
//   ui tree [--all]                 the focused window, one element per line with [ref=N]
//   ui click <ref>                  press/select/toggle the element (pointer click as fallback)
//   ui type <ref> <text> [--enter]  focus a text field, replace its text, optionally press Enter
// Refs number the elements in tree order, so they stay valid until the window changes:
// take a fresh `ui tree` after anything that changes the page.
import { Machine, AccessibilitySnapshot } from '@simular-ai/simulang-js'
import { execFileSync } from 'node:child_process'

const [command = 'tree', ...args] = process.argv.slice(2)
const machine = Machine.local()

function snapshot(): AccessibilitySnapshot {
  const window = machine.focusedWindow()
  if (!window) {
    console.error('Nothing is open on the screen yet. Run ui open <url> first (for example: ui open https://example.com), then ui tree.')
    process.exit(1)
  }
  return AccessibilitySnapshot.fromWindow(window)
}

// Lines a person would never act on: unnamed wrappers and Chromium's empty alert/badge nodes.
const noise = /^\s*- (alert|group|toolbar|section|generic|text|button|image|img|separator)( \[disabled\])? \[ref=\d+\]$/

function tree(all: boolean): string {
  const text = snapshot().print({ maxChars: 100 })
  return all ? text : text.split('\n').filter((line) => !noise.test(line)).join('\n')
}

function pointerClick(s: AccessibilitySnapshot, ref: number) {
  const [x, y] = s.clickablePoint(ref, true)
  execFileSync('xdotool', ['mousemove', String(Math.round(x)), String(Math.round(y)), 'click', '1'])
}

function ref(value: string | undefined): number {
  const n = Number(value)
  if (!Number.isInteger(n) || n < 0) {
    console.error(`Expected a ref number from ui tree, got "${value ?? ''}"`)
    process.exit(2)
  }
  return n
}

switch (command) {
  case 'tree':
    console.log(tree(args.includes('+all')))
    break
  case 'click': {
    const s = snapshot()
    const n = ref(args[0])
    try {
      s.click(n)
    } catch {
      pointerClick(s, n)
    }
    console.log(`Clicked ref ${n}. Run ui tree to see the result.`)
    break
  }
  case 'type': {
    const s = snapshot()
    const n = ref(args[0])
    const enter = args.includes('+enter')
    const text = args.slice(1).filter((a) => a !== '+enter').join(' ')
    pointerClick(s, n)
    execFileSync('xdotool', ['key', 'ctrl+a'])
    execFileSync('xdotool', ['type', '--delay', '20', text])
    if (enter) execFileSync('xdotool', ['key', 'Return'])
    console.log(`Typed into ref ${n}${enter ? ' and pressed Enter' : ''}. Run ui tree to see the result.`)
    break
  }
  default:
    console.error(`Unknown command "${command}". Use: ui open <url> | ui tree | ui click <ref> | ui type <ref> <text> [--enter] | ui key <keys> | ui screenshot`)
    process.exit(2)
}
