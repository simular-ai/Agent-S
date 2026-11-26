// The agent's browser verbs, run by /usr/local/bin/web. They drive the computer's Chromium
// through its debugging port (local to the computer only), like box's browser tools:
//   web open <url>               go to a page (opens Chromium if needed)
//   web snapshot [+all]          the page as a tree with [ref=eN] for every element you can use
//   web click <ref>              click an element
//   web type <ref> <text> [+enter]  fill a text field, optionally press Enter
//   web read                     the page's title, address and visible text
//   web key <key>                press a key in the page (Enter, Tab, Escape, PageDown, ...)
//   web back | web scroll [down|up] | web tabs
// Refs come from the latest snapshot of the page; take a new one after the page changes.
import { chromium } from '/usr/local/lib/sai/node_modules/playwright-core/index.mjs'

const CDP = 'http://127.0.0.1:9222'
const [command = 'snapshot', ...args] = process.argv.slice(2)
const flag = (name) => args.includes('+' + name)
const words = args.filter((a) => !a.startsWith('+'))

function fail(message, code = 1) {
  console.error(message)
  process.exit(code)
}

async function connect() {
  try {
    return await chromium.connectOverCDP(CDP, { timeout: 5000 })
  } catch {
    fail('The browser is not open. Run web open <url> first (for example: web open https://example.com).')
  }
}

// The tab the user sees: the last one that isn't a devtools or blank background page.
function currentPage(browser) {
  const pages = browser.contexts().flatMap((c) => c.pages()).filter((p) => !p.url().startsWith('devtools://'))
  return pages.at(-1)
}

async function settle(page) {
  await page.waitForLoadState('domcontentloaded', { timeout: 10000 }).catch(() => {})
  await page.waitForTimeout(800)
}

async function where(page) {
  return `Now at: ${await page.title().catch(() => '')} (${page.url()})`
}

// Wrappers with no name, and noise a person never acts on.
const noise = /^\s*- (generic|group|none|presentation)( \[ref=e\d+\])?:?$/
const MAX_LINES = 400

// While a dialog is open (a compose box, a sign-in popup), only it can be used: anything behind
// it is covered. Snapshots and refs then cover just the dialog, so the agent can't pick a
// look-alike button behind it.
async function scope(page) {
  const dialogs = page.locator('[role=dialog][aria-modal=true], dialog[open]')
  const count = await dialogs.count().catch(() => 0)
  for (let i = count - 1; i >= 0; i--) {
    const dialog = dialogs.nth(i)
    if (await dialog.isVisible().catch(() => false)) return { target: dialog, dialog: true }
  }
  return { target: page, dialog: false }
}

async function snapshot(page, all) {
  const { target, dialog } = await scope(page)
  const text = (dialog ? 'A dialog is open; only it is shown (web key Escape closes it).\n' : '')
    + (await target.ariaSnapshot({ mode: 'ai' }))
  let lines = text.split('\n').map((l) => l.replace(/ \[cursor=pointer\]/g, ''))
  if (!all) lines = lines.filter((l) => !noise.test(l))
  if (lines.length > MAX_LINES) {
    const more = lines.length - MAX_LINES
    lines = lines.slice(0, MAX_LINES)
    lines.push(`… ${more} more lines. Use web scroll down and web snapshot again, or web read for the text.`)
  }
  return lines.join('\n')
}

// Refs live in the page; a fresh snapshot (same page, same order) makes them resolvable here.
async function element(page, ref) {
  if (!/^e\d+$/.test(ref ?? '')) fail(`Expected a ref like e12 from web snapshot, got "${ref ?? ''}"`, 2)
  await (await scope(page)).target.ariaSnapshot({ mode: 'ai' })
  const locator = page.locator(`aria-ref=${ref}`)
  if ((await locator.count()) === 0) fail(`No element ${ref} on the page now. Run web snapshot to get fresh refs.`)
  return locator
}

async function main() {
  if (command === 'open') {
    let url = words[0]
    if (!url) fail('Usage: web open <url>', 2)
    if (!/^[a-z]+:\/\//i.test(url)) url = 'https://' + url
    const browser = await connect()
    const page = currentPage(browser) ?? (await browser.contexts()[0].newPage())
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 }).catch((e) => fail(`Could not open ${url}: ${e.message}`))
    // Show the page the agent works on, so the person watching the screen sees it.
    await page.bringToFront().catch(() => {})
    await settle(page)
    console.log(`${await where(page)}. Run web snapshot to see the page.`)
    return
  }

  const browser = await connect()
  const page = currentPage(browser)
  if (!page) fail('No page is open. Run web open <url> first.')

  switch (command) {
    case 'snapshot':
      console.log(await where(page))
      console.log(await snapshot(page, flag('all')))
      break
    case 'read': {
      const text = await page.evaluate(() => document.body?.innerText ?? '')
      const clipped = text.length > 8000 ? text.slice(0, 8000) + '\n… (more below; web scroll down then web read)' : text
      console.log(`${await where(page)}\n\n${clipped}`)
      break
    }
    case 'click': {
      const target = await element(page, words[0])
      await target.click({ timeout: 8000 }).catch((e) => {
        const covered = /intercepts pointer events|not visible|outside of the viewport/.test(e.message)
        fail(`FAILED: could not click ${words[0]}${covered ? ' because something covers it (often an open dialog)' : ''}. Nothing was clicked. Run web snapshot and pick a ref from what it shows.`)
      })
      await settle(page)
      console.log(`Clicked ${words[0]}. ${await where(page)}. Run web snapshot to see the result.`)
      break
    }
    case 'type': {
      const target = await element(page, words[0])
      const text = words.slice(1).join(' ')
      await target.fill(text, { timeout: 10000 }).catch(async () => {
        await target.click({ timeout: 5000 })
        await page.keyboard.type(text, { delay: 20 })
      })
      if (flag('enter')) {
        await target.press('Enter')
        await settle(page)
      }
      console.log(`Typed into ${words[0]}${flag('enter') ? ' and pressed Enter' : ''}. ${await where(page)}`)
      break
    }
    case 'key':
      if (!words[0]) fail('Usage: web key <key>   (Enter, Tab, Escape, PageDown, Control+a, ...)', 2)
      await page.keyboard.press(words[0])
      await settle(page)
      console.log(`Pressed ${words[0]}. ${await where(page)}`)
      break
    case 'back':
      await page.goBack({ timeout: 15000 }).catch(() => {})
      await settle(page)
      console.log(await where(page))
      break
    case 'scroll':
      await page.mouse.wheel(0, words[0] === 'up' ? -700 : 700)
      await page.waitForTimeout(600)
      console.log(`Scrolled ${words[0] === 'up' ? 'up' : 'down'}. Run web snapshot to see what is visible.`)
      break
    case 'tabs': {
      const pages = browser.contexts().flatMap((c) => c.pages())
      for (const [i, p] of pages.entries()) console.log(`${i + 1}. ${await p.title()} (${p.url()})`)
      break
    }
    default:
      fail(`Unknown command "${command}". Use: web open <url> | web snapshot | web click <ref> | web type <ref> <text> [--enter] | web read | web key <key> | web back | web scroll [down|up] | web tabs`, 2)
  }
}

// Disconnect without closing the browser.
main().then(() => process.exit(0), (e) => fail(e.message ?? String(e)))
