#!/usr/bin/env python3
"""Uso: python peer_lock_check.py <db-path> <peer> <mode> [timeout_ms]
mode = 'hold' (acquire, espera 2s, libera) | 'contend' (tenta acquire, imprime resultado)
"""
import json
import sys
import time

from gui_agents.s3.coordination.peer_lock import PeerLock


def main():
    db_path, peer, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    lock = PeerLock(db_path)
    if mode == "hold":
        result = lock.acquire(peer, "hold-task", timeout_ms=5000)
        print(json.dumps({"step": "acquire", **result}))
        time.sleep(2)
        lock.release(peer)
        print(json.dumps({"step": "release", "ok": True}))
    elif mode == "contend":
        timeout_ms = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
        result = lock.acquire(peer, "contend-task", timeout_ms=timeout_ms)
        print(json.dumps({"step": "acquire", **result}))
    else:
        raise ValueError(f"modo desconhecido: {mode}")
    lock.close()


if __name__ == "__main__":
    main()
