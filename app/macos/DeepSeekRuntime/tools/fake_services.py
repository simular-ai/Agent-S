"""Synthetic model and MCP servers for the DeepSeek runtime integration test.

Prints {"model": <port>, "mcp": <port>} on one line, then serves until stdin closes.
The model streams OpenAI-compatible chat completions: a prompt containing "use tool" calls the
first tool, "go slow" streams slowly, and anything else is echoed. The MCP server answers
streamable-HTTP JSON-RPC with one `mac_command` tool.
"""
import http.server
import json
import sys
import threading
import time


class Model(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = request['messages']
        replied = [i for i, m in enumerate(messages) if m['role'] == 'assistant']
        recent = messages[replied[-1] + 1:] if replied else messages
        text = json.dumps([m.get('content') for m in recent if m['role'] == 'user'])
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()

        def chunk(delta, finish=None):
            payload = {'id': 'synthetic', 'object': 'chat.completion.chunk', 'created': 0, 'model': request['model'],
                       'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
            self.wfile.write(b'data: ' + json.dumps(payload).encode() + b'\n\n')
            self.wfile.flush()

        if messages[-1]['role'] == 'user' and 'use tool' in text:
            name = request['tools'][0]['function']['name']
            chunk({'role': 'assistant', 'tool_calls': [{'index': 0, 'id': 'call_1', 'type': 'function',
                   'function': {'name': name, 'arguments': json.dumps({'command': 'echo synthetic'})}}]})
            chunk({}, 'tool_calls')
        elif messages[-1]['role'] == 'user' and 'go slow' in text:
            for word in ['Working ', 'slowly ', 'now.']:
                chunk({'role': 'assistant', 'content': word})
                time.sleep(1.0)
            chunk({}, 'stop')
        else:
            last = messages[-1].get('content')
            reply = 'Echo: ' + (last if isinstance(last, str) else 'tool result')
            for word in reply.split(' '):
                chunk({'role': 'assistant', 'content': word + ' '})
            chunk({}, 'stop')
        self.wfile.write(b'data: [DONE]\n\n')


class MCP(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        message = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if 'id' not in message:
            self.send_response(202)
            self.end_headers()
            return
        reply = {'jsonrpc': '2.0', 'id': message['id']}
        method = message['method']
        if method == 'initialize':
            reply['result'] = {'protocolVersion': message['params'].get('protocolVersion', '2025-06-18'),
                               'capabilities': {'tools': {}}, 'serverInfo': {'name': 'Synthetic', 'version': '1'}}
        elif method == 'tools/list':
            reply['result'] = {'tools': [{'name': 'mac_command', 'description': 'Run a command.',
                                          'inputSchema': {'type': 'object', 'properties': {'command': {'type': 'string'}},
                                                          'required': ['command']}}]}
        elif method == 'tools/call':
            reply['result'] = {'content': [{'type': 'text', 'text': 'synthetic output'}], 'isError': False}
        else:
            reply['result'] = {}
        body = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    servers = [http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler) for handler in (Model, MCP)]
    for server in servers:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    print(json.dumps({'model': servers[0].server_port, 'mcp': servers[1].server_port}), flush=True)
    sys.stdin.read()


if __name__ == '__main__':
    main()
