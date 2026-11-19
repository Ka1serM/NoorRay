#! Serves a web build on 127.0.0.1 with the headers that threads need: python3 tools/serve_web.py <build>/<app folder> [port]
import functools
import http.server
import os
import sys


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_head(self):
        if not os.path.exists(self.translate_path(self.path)): self.path = "/index.html"
        return super().send_head()


directory = sys.argv[1]
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
server = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=directory))
print(f"Open http://127.0.0.1:{port}/", flush=True)
server.serve_forever()
