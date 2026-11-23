import Foundation

/// The guest's screen tools, installed by the app on every start (so existing computers get
/// updates too): `web` for websites, and `ui` to launch apps, inspect the focused window's
/// accessibility tree with refs (through simulang), click and type by ref, press keys and take
/// screenshots. Clicking a named element is far easier for a small local model than guessing
/// pixel positions from a screenshot.
public enum GuestTools {
    public static let uiPath = "/usr/local/bin/ui"
    public static let uiScriptPath = "/usr/local/lib/sai/ui.ts"
    static let desktopLauncherPath = "/usr/local/lib/sai/launch-desktop-app"

    /// `/usr/local/bin/ui`: sets up the display and accessibility bus, then runs a verb.
    public static let ui = #"""
#!/bin/bash
# The agent's way to use the screen: open a site, read the focused window as an accessibility
# tree with refs, click and type by ref, press keys, or take a screenshot.
#   ui open <url>                  open Chromium (with its accessibility tree switched on)
#   ui launch <app> [args...]      launch an installed desktop app on the computer
#   ui tree [--all]                list the focused window's elements, each with [ref=N]
#   ui click <ref>                 click an element
#   ui type <ref> <text> [--enter] type into a text field (replacing its text)
#   ui key <keys...>               press keys, e.g. ui key ctrl+l   ui key Return
#   ui screenshot                  save a half-size screenshot and print its path
if [ "${1:-}" = launch ]; then
  shift
  exec \#(desktopLauncherPath) "$@"
fi
export DISPLAY="${DISPLAY:-:0}"

# Commands come in over SSH, outside the desktop session: borrow its D-Bus, where the
# accessibility bus lives.
if [ -z "$DBUS_SESSION_BUS_ADDRESS" ]; then
  pid=$(pgrep -x openbox | head -1)
  if [ -n "$pid" ]; then
    export "$(tr '\0' '\n' < "/proc/$pid/environ" | grep '^DBUS_SESSION_BUS_ADDRESS=')"
  fi
fi
# Apps only publish their accessibility trees while this is on.
dbus-send --session --dest=org.a11y.Bus /org/a11y/bus org.freedesktop.DBus.Properties.Set \
  string:org.a11y.Status string:IsEnabled variant:boolean:true >/dev/null 2>&1

case "$1" in
  open)
    # One browser for both tools: web opens Chromium with its debugging port and accessibility on.
    shift
    exec web open "$@"
    ;;
  key)
    shift
    xdotool key "$@" && echo "Pressed $*."
    ;;
  screenshot)
    scrot -o -t 50 /home/sai/work/screen.png && echo /home/sai/work/screen-thumb.png
    ;;
  *)
    # `simulang run` would take --flags as its own; hand them to the script as plain words.
    args=()
    for a in "$@"; do case "$a" in --enter) args+=(+enter) ;; --all) args+=(+all) ;; *) args+=("$a") ;; esac; done
    simulang run /usr/local/lib/sai/ui.ts "${args[@]}" 2> >(grep -v -e OPENROUTER -e '^\[info\]' >&2)
    ;;
esac
"""#

    /// The simulang script behind `ui tree`, `ui click` and `ui type`.
    public static let uiScript = #"""
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
    console.error('Nothing is open on the screen yet. Start an app with ui launch <app> or open a website with ui open <url>, then run ui tree.')
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
    console.error(`Unknown command "${command}". Use: ui open <url> | ui launch <app> [args...] | ui tree | ui click <ref> | ui type <ref> <text> [--enter] | ui key <keys> | ui screenshot`)
    process.exit(2)
}
"""#

    /// Ask the desktop's user manager to start an app independently of the shell tool.
    static let desktopLauncher = #"""
#!/bin/bash
if [ "$#" -eq 0 ] || [ -z "$1" ]; then
  echo "Use: ui launch <desktop-id, executable, or absolute .desktop file> [args...]" >&2
  exit 2
fi
app=$1
shift

find_desktop_file() {
  local name="${1%.desktop}.desktop" root directory file relative
  local -a data_dirs
  IFS=: read -r -a data_dirs <<< "${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"
  for root in "${XDG_DATA_HOME:-$HOME/.local/share}" "${data_dirs[@]}" \
      "$HOME/.local/share/flatpak/exports/share" /var/lib/flatpak/exports/share; do
    [[ $root == /* ]] || continue
    directory="$root/applications"
    file="$directory/$name"
    if [[ -f $file && -r $file ]]; then
      printf '%s\n' "$file"
      return 0
    fi
    # Desktop IDs turn slashes in paths below applications/ into dashes.
    [[ $name == *-* && -d $directory ]] || continue
    while IFS= read -r -d '' file; do
      relative=${file#"$directory/"}
      if [[ ${relative//\//-} == "$name" && -f $file && -r $file ]]; then
        printf '%s\n' "$file"
        return 0
      fi
    done < <(find "$directory" -mindepth 2 -name '*.desktop' -print0 2>/dev/null)
  done
  return 1
}

desktop_file=
case "$app" in
  /*.desktop)
    if [[ ! -f $app || ! -r $app ]]; then
      printf 'Desktop file is unavailable: %s\n' "$app" >&2
      exit 1
    fi
    desktop_file=$app
    ;;
  */*) ;;
  *) desktop_file=$(find_desktop_file "$app") ;;
esac

if [[ -n $desktop_file ]]; then
  if ! gio=$(type -P -- gio); then
    echo "Desktop launching needs gio. Start the computer with the current Agent S build to install it." >&2
    exit 1
  fi
  command=("$gio" launch -- "$desktop_file" "$@")
else
  executable=$app
  if [[ $app != */* ]]; then
    executable=$(type -P -- "$app")
  fi
  if [[ ! -f $executable || ! -x $executable ]]; then
    printf 'No installed desktop entry or executable found for %s.\n' "$app" >&2
    exit 1
  fi
  command=("$executable" "$@")
fi
if [[ ${command[0]} != /* ]]; then command[0]="$PWD/${command[0]}"; fi

uid=$(id -u)
runtime="${XDG_RUNTIME_DIR:-/run/user/$uid}"
display=:0
authority="$HOME/.Xauthority"
desktop_bus=
desktop=$(pgrep -u "$uid" -xo openbox)
if [[ -n $desktop && -r /proc/$desktop/environ ]]; then
  while IFS= read -r -d '' setting; do
    case "$setting" in
      DISPLAY=*) display=${setting#*=} ;;
      XAUTHORITY=*) authority=${setting#*=} ;;
      DBUS_SESSION_BUS_ADDRESS=*) desktop_bus=${setting#*=} ;;
    esac
  done < "/proc/$desktop/environ"
fi

# The user manager must use its own bus. The desktop bus is for the app only.
# A service (rather than a scope) starts outside the harness's cgroup.
# Gio and other launchers can exit while their child keeps displaying the app.
options=(--user --quiet --collect --service-type=exec --expand-environment=no
  --property=ExitType=cgroup --property=StandardInput=null
  --description="Agent S desktop app" --working-directory="$PWD"
  --setenv="XDG_RUNTIME_DIR=$runtime" --setenv="DISPLAY=$display" --setenv="XAUTHORITY=$authority")
if [[ -n $desktop_bus ]]; then options+=(--setenv="DBUS_SESSION_BUS_ADDRESS=$desktop_bus"); fi
env -u DBUS_SESSION_BUS_ADDRESS XDG_RUNTIME_DIR="$runtime" systemd-run "${options[@]}" -- "${command[@]}" || exit $?
printf 'Requested launch of %s. Use ui tree or ui screenshot to check the desktop.\n' "$app"
"""#

    public static let webPath = "/usr/local/bin/web"
    public static let webScriptPath = "/usr/local/lib/sai/web.mjs"
    private static let browserServiceName = "agent-s-browser.service"
    private static let browserLauncherPath = "/usr/local/lib/sai/start-browser"
    /// Pinned so every computer runs the same browser driver.
    public static let playwrightVersion = "1.65.1"

    /// The user manager starts Chromium outside the harness's per-turn process scope.
    static let browserService = """
        [Unit]
        Description=Agent S Chromium
        [Service]
        Type=exec
        ExecStart=\(browserLauncherPath)
        """

    static let browserLauncher = #"""
#!/bin/bash
# Use the existing guest desktop bus and X display so Chromium is visible and accessible.
desktop=$(pgrep -u "$(id -u)" -xo openbox)
if [ -n "$desktop" ]; then
  while IFS= read -r -d '' setting; do
    case "$setting" in DISPLAY=*|XAUTHORITY=*|DBUS_SESSION_BUS_ADDRESS=*) export "$setting" ;; esac
  done < "/proc/$desktop/environ"
fi
export DISPLAY="${DISPLAY:-:0}" XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" ACCESSIBILITY_ENABLED=1

# Chromium runs as the same guest user and continues using its existing profile.
exec /usr/bin/chromium --no-first-run --start-maximized \
  --force-renderer-accessibility --remote-debugging-port=9222 --remote-debugging-address=127.0.0.1 \
  about:blank
"""#

    /// `/usr/local/bin/web`: the agent's browser, like box's browser tools. Drives the computer's
    /// Chromium through its debugging port (on the computer's loopback only) with Playwright.
    public static let web = #"""
#!/bin/bash
# The agent's browser: drives the computer's Chromium through its debugging port.
#   web open <url>                   go to a page (opens Chromium if needed)
#   web snapshot                     the page with [ref=eN] for every element you can use
#   web click <ref>                  click an element
#   web type <ref> <text> [--enter]  fill a text field, optionally press Enter
#   web read                         the page's visible text
#   web key <key> | web back | web scroll [down|up] | web tabs
browser_ready() {
  curl -fs --max-time 0.5 http://127.0.0.1:9222/json/version >/dev/null 2>&1
}

wait_for_browser() {
  for _ in {1..30}; do
    if browser_ready; then return 0; fi
    sleep 0.25
  done
  return 1
}

if ! browser_ready; then
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  if [ "${1:-}" = open ]; then
    if ! systemctl --user start \#(browserServiceName); then
      echo "Couldn't start the browser on the computer." >&2
      exit 1
    fi
  elif ! systemctl --user is-active --quiet \#(browserServiceName); then
    echo "The browser is not open. Run web open <url> first (for example: web open https://example.com)." >&2
    exit 1
  fi
  if ! wait_for_browser; then
    echo "Couldn't connect to Chromium's browser tools. Check the computer screen and try again." >&2
    exit 1
  fi
fi

args=()
for a in "$@"; do case "$a" in --enter) args+=(+enter) ;; --all) args+=(+all) ;; *) args+=("$a") ;; esac; done
exec node /usr/local/lib/sai/web.mjs "${args[@]}"
"""#

    /// The Playwright script behind `web`.
    public static let webScript = #"""
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
"""#

    /// Refresh the guest scripts and unit definitions without changing the desktop session.
    public static var installScriptsCommand: String {
        func put(_ content: String, _ path: String, _ mode: String) -> String {
            let data = Data(content.utf8).base64EncodedString()
            return "sudo mkdir -p \((path as NSString).deletingLastPathComponent) && echo \(data) | base64 -d | sudo tee \(path) >/dev/null && sudo chmod \(mode) \(path)"
        }
        return [
            put(ui, uiPath, "755"), put(uiScript, uiScriptPath, "644"),
            put(desktopLauncher, desktopLauncherPath, "755"),
            put(web, webPath, "755"), put(webScript, webScriptPath, "644"),
            put(browserLauncher, browserLauncherPath, "755"),
            put(browserService, "/etc/systemd/user/\(browserServiceName)", "644"),
            #"XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" systemctl --user daemon-reload"#,
        ].joined(separator: " && ")
    }

    public static var installCommand: String {
        [GuestSetup.ensureNodeCommand(asRoot: false), GuestSetup.ensureBaseHarnessesCommand(asRoot: false),
         installScriptsCommand, desktopLaunchDependencies, playwright, clipboard, terminal, installPinnedDeepSeek]
            .joined(separator: " && ")
    }

    private static let desktopLaunchDependencies =
        "(command -v gio >/dev/null || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q libglib2.0-bin >/dev/null 2>&1)"

    /// The terminal used to sign in to harness accounts from the guest desktop.
    private static let terminal = "(command -v xterm >/dev/null || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q xterm >/dev/null 2>&1)"

    /// A completed check requests an update only when the installed versions are unsuitable.
    public static let deepSeekUpdateRequiredStatus: Int32 = 65

    public static var deepSeekReadyCommand: String {
        """
        /usr/bin/timeout --kill-after=2s 8s /bin/bash -s <<'AGENT_S_TOOL_CHECK'
        if \(GuestSetup.nodeReadyCommand) && \(deepSeekVersionCheck); then
          exit 0
        fi
        exit \(deepSeekUpdateRequiredStatus)
        AGENT_S_TOOL_CHECK
        """
    }

    /// Repairs the runtime on an existing guest without restarting its desktop.
    public static var installDeepSeekCommand: String {
        "\(GuestSetup.ensureNodeCommand(asRoot: false)) && \(installPinnedDeepSeek)"
    }

    private static var deepSeekVersionCheck: String {
        #"(dsh_version=$(dsh --version 2>/dev/null) && [ "$dsh_version" = '\#(DeepSeekCommand.version)' ])"#
    }

    private static var installPinnedDeepSeek: String {
        "(\(deepSeekVersionCheck) || (sudo npm install -g --engine-strict @deepseek-ai/dsh@\(DeepSeekCommand.version)"
            + " && \(deepSeekVersionCheck)))"
    }

    /// Playwright's library (no bundled browsers) for `web`, installed once per version.
    static var playwright: String {
        "(grep -qs '\"version\": \"\(playwrightVersion)\"' /usr/local/lib/sai/node_modules/playwright-core/package.json"
            + " || (cd /usr/local/lib/sai && sudo npm install --no-audit --no-fund --silent playwright-core@\(playwrightVersion) >/dev/null 2>&1))"
    }

    /// Clipboard sharing for computers set up before it existed: install spice-vdagent once and
    /// start it with the desktop. It must run inside the desktop's login session (spice-vdagentd
    /// ignores agents from other sessions, such as this SSH one), so when it isn't running the
    /// desktop session is restarted and Openbox's autostart starts it.
    static var clipboard: String {
        let autostart = "/home/sai/.config/openbox/autostart"
        return [
            "(command -v spice-vdagent >/dev/null || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q spice-vdagent >/dev/null 2>&1)",
            "(grep -q spice-vdagent \(autostart) 2>/dev/null || { mkdir -p $(dirname \(autostart)) && echo spice-vdagent >> \(autostart); })",
            "(pgrep -x spice-vdagent >/dev/null || sudo systemctl restart getty@tty1)",
            "true",
        ].joined(separator: " && ")
    }
}
