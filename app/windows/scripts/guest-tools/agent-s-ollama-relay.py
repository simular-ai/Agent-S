#!/usr/bin/python3
# The guest half of the Ollama relay. The WSL guest can't reach the Windows host's 127.0.0.1, so this
# listens on the guest's own 127.0.0.1:11434 (and ::1) and multiplexes each connection over stdio,
# which reaches Agent S through wsl.exe; the app connects it to Ollama on the host.
# Frame: type (1 byte), connection id (uint32 BE), length (uint32 BE), payload.
# Types: 1 open (guest to host), 2 data (both ways), 3 close (both ways).
import os, selectors, socket, struct, sys

OPEN, DATA, CLOSE = 1, 2, 3
PORT = 11434
sel = selectors.DefaultSelector()
out = sys.stdout.buffer
conns = {}
next_id = 1


def send(kind, ident, payload=b""):
    out.write(struct.pack(">BII", kind, ident, len(payload)) + payload)
    out.flush()


def drop(ident, tell=True):
    sock = conns.pop(ident, None)
    if sock is None:
        return
    try:
        sel.unregister(sock)
    except (KeyError, ValueError):
        pass
    sock.close()
    if tell:
        send(CLOSE, ident)


listeners = []
for family, address in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
    try:
        server = socket.socket(family, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((address, PORT))
        server.listen(64)
        server.setblocking(False)
        sel.register(server, selectors.EVENT_READ, "listen")
        listeners.append(server)
    except OSError:
        pass
if not listeners:
    sys.stderr.write("agent-s-ollama-relay: port %d is busy\n" % PORT)
    sys.exit(1)

os.set_blocking(0, False)
sel.register(0, selectors.EVENT_READ, "stdin")
pending = b""

while True:
    for key, _ in sel.select():
        if key.data == "listen":
            try:
                sock, _ = key.fileobj.accept()
            except OSError:
                continue
            ident = next_id
            next_id = next_id % 0xFFFFFFFF + 1
            sock.setblocking(True)
            conns[ident] = sock
            sel.register(sock, selectors.EVENT_READ, ident)
            send(OPEN, ident)
        elif key.data == "stdin":
            try:
                chunk = os.read(0, 65536)
            except BlockingIOError:
                continue
            if not chunk:
                sys.exit(0)
            pending += chunk
            while len(pending) >= 9:
                kind, ident, length = struct.unpack(">BII", pending[:9])
                if len(pending) < 9 + length:
                    break
                payload, pending = pending[9:9 + length], pending[9 + length:]
                if kind == DATA and ident in conns:
                    try:
                        conns[ident].sendall(payload)
                    except OSError:
                        drop(ident)
                elif kind == CLOSE:
                    sock = conns.get(ident)
                    if sock is not None:
                        # Let the client read what was sent, then close.
                        try:
                            sock.shutdown(socket.SHUT_WR)
                        except OSError:
                            pass
                        drop(ident, tell=False)
        else:
            ident = key.data
            sock = conns.get(ident)
            if sock is None:
                continue
            try:
                data = sock.recv(65536)
            except OSError:
                data = b""
            if data:
                send(DATA, ident, data)
            else:
                drop(ident)
