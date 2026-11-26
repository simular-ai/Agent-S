"""Reduce an installed dsh tree to the packages Agent S's profile loads.

Usage: prune.py <node_modules> <packages.txt> <macho-list-output>

Removes packages missing from packages.txt, then files no runtime reads: TypeScript sources,
declarations, source maps, documentation, and other platforms' prebuilt binaries. Writes the
paths of the remaining Mach-O files, relative to node_modules, for signing.
"""
import os
import shutil
import sys

JS = ('.js', '.mjs', '.cjs', '.json', '.node', '.wasm')
DROP_FILES = ('.map', '.md', '.d.ts', '.d.mts', '.d.cts', '.tsbuildinfo')
DROP_DIRS = ('docs', 'test', 'tests', '__tests__', 'example', 'examples')
MACHO = {b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca'}


def prune_packages(root, keep, prefix=''):
    base = os.path.join(root, prefix) if prefix else root
    modules = os.path.join(base, 'node_modules') if prefix else base
    if not os.path.isdir(modules):
        return
    for name in sorted(os.listdir(modules)):
        if name.startswith('.'):
            path = os.path.join(modules, name)
            shutil.rmtree(path) if os.path.isdir(path) and not os.path.islink(path) else os.remove(path)
            continue
        scope = os.path.join(modules, name)
        names = [name + '/' + child for child in os.listdir(scope)] if name.startswith('@') else [name]
        for package in names:
            rel = prefix + '/node_modules/' + package if prefix else package
            if rel not in keep:
                shutil.rmtree(os.path.join(modules, package))
            else:
                prune_packages(root, keep, rel)
        if name.startswith('@') and not os.listdir(scope):
            os.rmdir(scope)


def has_runtime_code(folder):
    return any(f.endswith(JS) for _, _, files in os.walk(folder) for f in files)


def strip_files(root):
    for folder, dirs, files in os.walk(root):
        package = os.path.isfile(os.path.join(folder, 'package.json'))
        for name in list(dirs):
            path = os.path.join(folder, name)
            if package and (name in DROP_DIRS or (name == 'src' and not has_runtime_code(path))):
                shutil.rmtree(path)
                dirs.remove(name)
            elif name == 'prebuilds':
                for platform in os.listdir(path):
                    if platform != 'darwin-arm64':
                        shutil.rmtree(os.path.join(path, platform))
        for name in files:
            if name.endswith(DROP_FILES) and not name.lower().startswith(('license', 'licence', 'notice')):
                os.remove(os.path.join(folder, name))


def mach_o_files(root):
    found = []
    for folder, _, files in os.walk(root):
        for name in files:
            path = os.path.join(folder, name)
            if os.path.islink(path):
                continue
            with open(path, 'rb') as handle:
                if handle.read(4) in MACHO:
                    found.append(os.path.relpath(path, root))
    return sorted(found)


def main():
    root, packages, output = sys.argv[1:4]
    keep = {line.strip() for line in open(packages) if line.strip() and not line.startswith('#')}
    prune_packages(os.path.realpath(root), keep)
    strip_files(root)
    with open(output, 'w') as handle:
        handle.writelines(path + '\n' for path in mach_o_files(root))


if __name__ == '__main__':
    main()
