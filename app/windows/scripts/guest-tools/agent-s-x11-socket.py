#!/usr/bin/python3
# Gives clients that only look for /tmp/.X11-unix/X<n> (such as simulang) a way to the Private computer's
# display. WSLg mounts /tmp/.X11-unix read-only, so Agent S's Xvfb listens only on the abstract socket
# @/tmp/.X11-unix/X<n>. `ui` runs this inside its own mount namespace, over a private writable
# /tmp/.X11-unix, to proxy a filesystem socket to the abstract one. It exits with its parent.
import os, socket, sys, threading

display = sys.argv[1]
path = "/tmp/.X11-unix/X" + display
abstract = "\0/tmp/.X11-unix/X" + display
parent = os.getppid()


def pump(source, target):
    try:
        while True:
            data = source.recv(65536)
            if not data:
                break
            target.sendall(data)
    except OSError:
        pass
    for sock in (source, target):
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def serve(client):
    upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        upstream.connect(abstract)
    except OSError:
        client.close()
        return
    threading.Thread(target=pump, args=(client, upstream), daemon=True).start()
    threading.Thread(target=pump, args=(upstream, client), daemon=True).start()


def watch_parent():
    while True:
        threading.Event().wait(1)
        if os.getppid() != parent:
            os._exit(0)


server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    os.unlink(path)
except FileNotFoundError:
    pass
server.bind(path)
server.listen(16)
threading.Thread(target=watch_parent, daemon=True).start()
print("ready", flush=True)
while True:
    connection, _ = server.accept()
    serve(connection)
