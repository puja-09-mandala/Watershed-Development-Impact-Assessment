"""Static server for the Rajapeta 5-page web GIS.

Clean routes (no home/dashboard page; the app opens directly into NDVI):
    /ndvi        -> web/ndvi.html
    /lst         -> web/lst.html
    /lulc        -> web/lulc.html
    /llm-report  -> web/llm.html
    /charts      -> web/charts.html
    /            -> /ndvi
"""

from __future__ import annotations
import http.server
import json
import socketserver
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
PORT = 8080

# Clean route -> page file.
ROUTES = {
    "/ndvi": "ndvi.html",
    "/lst": "lst.html",
    "/lulc": "lulc.html",
    "/llm-report": "llm.html",
    "/charts": "charts.html",
}
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0].split("#", 1)[0]

        # API: Mistral key status (boolean only — the key never leaves the server).
        if path == "/api/llm/status":
            from src.llm_interpreter import mistral_available

            self._send_json(
                {
                    "key_configured": mistral_available(),
                    "endpoint": "/api/llm/interpret",
                }
            )
            return

        # Clean routes and root (opens directly into NDVI).
        if path in ROUTES:
            self.path = f"/web/{ROUTES[path]}"
        elif path in ("/", "/index.html"):
            self.path = "/web/ndvi.html"
        elif path.rstrip("/") in ROUTES:
            self.path = f"/web/{ROUTES[path.rstrip('/')]}"
        # Relative assets referenced by clean-URL pages (css/js/data at root).
        elif path.startswith(("/css/", "/js/", "/data/", "/figures/")):
            self.path = "/web" + path
        elif path.startswith("/web/"):
            pass
        return super().do_GET()

    def do_POST(self):
        path = self.path.split("?", 1)[0].split("#", 1)[0]

        # Allow user to save Mistral API key from the browser UI.
        if path == "/api/llm/setkey":
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                key = str(body.get("key", "")).strip()
                if not key:
                    self._send_json({"error": "No key provided."}, status=400)
                    return
                env_path = ROOT / ".env"
                lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True) if env_path.exists() else []
                new_lines = []
                replaced = False
                for line in lines:
                    if line.startswith("MISTRAL_API_KEY="):
                        new_lines.append(f"MISTRAL_API_KEY={key}\n")
                        replaced = True
                    else:
                        new_lines.append(line)
                if not replaced:
                    new_lines.append(f"MISTRAL_API_KEY={key}\n")
                env_path.write_text("".join(new_lines), encoding="utf-8")
                # Reload env so the running server picks it up immediately.
                import os
                os.environ["MISTRAL_API_KEY"] = key
                self._send_json({"ok": True, "message": "Key saved to .env and active for this session."})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return

        if path != "/api/llm/interpret":
            self._send_json({"error": "unknown endpoint"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)

            from config import PATHS, STUDY_AREA
            from src.llm_interpreter import interpret, mistral_available

            stats_path = PATHS["web_data"] / "stats.json"
            if not stats_path.exists():
                self._send_json(
                    {"error": "stats.json not found — run `python main.py` first."},
                    status=400,
                )
                return
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            payload = {
                "rasters": stats.get("rasters", []),
                "cnn_lstm": stats.get("cnn_lstm", {}),
                "lulc_note": stats.get("lulc_note", ""),
                "change_summaries": stats.get("change_summaries", {}),
                "lulc_areas": stats.get("lulc_areas", {}),
                "study_area": stats.get("study_area", STUDY_AREA),
            }
            key_configured = mistral_available()
            text = interpret(payload)
            (PATHS["web_data"] / "llm_interpretation.md").write_text(text, encoding="utf-8")
            self._send_json(
                {
                    "text": text,
                    "source": "mistral" if key_configured else "fallback",
                    "key_configured": key_configured,
                }
            )
        except Exception as exc:  # noqa: BLE001 — API surface must not crash
            self._send_json({"error": str(exc)}, status=500)
    def end_headers(self):
        # Always serve fresh data during development.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # Quiet server log.
def main() -> None:
    print("Routes: /ndvi | /lst | /lulc | /llm-report | /charts")
    socketserver.TCPServer.allow_reuse_address = True
    # Windows can reserve 8080 (Hyper-V excluded port ranges), making
    # bind() fail with WinError 10013. Fall back to nearby free ports.
    candidates = [PORT] + list(range(8081, 8100)) + [8000, 5500]
    httpd = None
    chosen = PORT
    last_err: OSError | None = None
    for port in candidates:
        try:
            httpd = socketserver.TCPServer(("127.0.0.1", port), Handler)
            chosen = port
            break
        except OSError as exc:
            last_err = exc
    if httpd is None:
        raise last_err  # type: ignore[misc]

    url = f"http://127.0.0.1:{chosen}"
    print(f"Serving Rajapeta Web GIS at {url}/ndvi")
    try:
        webbrowser.open(f"{url}/ndvi")
    except Exception:
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        httpd.server_close()
if __name__ == "__main__":
    main()