// Records every file under node_modules that the runtime imports or reads, for packages.py.
// Loaded with `node --import` during `scripts/vendor-deepseek.sh --trace`; never shipped.
import { registerHooks, syncBuiltinESMExports } from 'node:module'
import fs from 'node:fs'

const output = process.env.AGENT_S_TRACE
const seen = new Set()
const record = value => {
  let path = value
  if (path instanceof URL) path = path.pathname
  else if (Buffer.isBuffer(path)) path = path.toString()
  if (typeof path !== 'string') return
  if (path.startsWith('file://')) path = new URL(path).pathname
  if (!path.includes('/node_modules/') || seen.has(path)) return
  seen.add(path)
  fs.appendFileSync(output, path + '\n')
}

registerHooks({ resolve(specifier, context, next) { const result = next(specifier, context); record(result.url); return result } })

const wrap = original => {
  const wrapped = function (path, ...rest) { record(path); return original.call(this, path, ...rest) }
  Object.assign(wrapped, original)
  if (original.native) wrapped.native = function (path, ...rest) { record(path); return original.native.call(this, path, ...rest) }
  return wrapped
}
for (const name of ['readFileSync', 'statSync', 'lstatSync', 'existsSync', 'readdirSync', 'openSync', 'realpathSync', 'accessSync',
  'readFile', 'stat', 'lstat', 'readdir', 'open', 'access', 'realpath']) fs[name] = wrap(fs[name])
for (const name of ['readFile', 'stat', 'lstat', 'readdir', 'open', 'access', 'realpath']) fs.promises[name] = wrap(fs.promises[name])
syncBuiltinESMExports()
