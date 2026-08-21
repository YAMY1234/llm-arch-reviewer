#!/usr/bin/env python3
"""Serve the static viewer plus allowlisted raw traces for Perfetto handoff.

The ``/__trace__`` endpoint never accepts a path.  It resolves an exact base
file name plus SHA256 against trace roots selected at server startup, which
keeps the viewer from becoming an arbitrary local-file server.
"""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ViewerHandler(SimpleHTTPRequestHandler):
    trace_index: dict[str, list[Path]] = {}
    digest_cache: dict[Path, str] = {}

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path != "/__trace__":
            super().do_GET()
            return
        query = parse_qs(parsed.query)
        file_name = (query.get("file") or [""])[0]
        expected_sha256 = (query.get("sha256") or [""])[0]
        if (
            not file_name
            or Path(file_name).name != file_name
            or not file_name.endswith(".trace.json.gz")
            or len(expected_sha256) != 64
        ):
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid trace identity")
            return
        selected = None
        for path in self.trace_index.get(file_name, []):
            actual = self.digest_cache.get(path)
            if actual is None:
                actual = sha256_file(path)
                self.digest_cache[path] = actual
            if actual == expected_sha256:
                selected = path
                break
        if selected is None:
            self.send_error(HTTPStatus.NOT_FOUND, "allowlisted trace not found")
            return
        size = selected.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/gzip")
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "private, no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        with selected.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                self.wfile.write(chunk)


def build_trace_index(roots: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for root in roots:
        resolved_root = root.resolve()
        if not resolved_root.is_dir():
            raise FileNotFoundError(f"trace root is not a directory: {resolved_root}")
        for candidate in resolved_root.rglob("*.trace.json.gz"):
            resolved = candidate.resolve()
            if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
                continue
            index.setdefault(resolved.name, []).append(resolved)
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--docs-root", type=Path, default=REPO_ROOT / "docs")
    parser.add_argument(
        "--trace-root",
        type=Path,
        action="append",
        help=(
            "allowlisted raw-trace root; repeatable (defaults to "
            "../current/qwen40-*/raw when present)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    current_root = REPO_ROOT.parent / "current"
    default_trace_roots = sorted(current_root.glob("qwen40-*/raw"))
    trace_roots = args.trace_root or default_trace_roots
    ViewerHandler.trace_index = build_trace_index(trace_roots)
    handler = partial(ViewerHandler, directory=str(args.docs_root.resolve()))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    trace_count = sum(len(paths) for paths in ViewerHandler.trace_index.values())
    print(f"viewer: http://{args.host}:{args.port}/viewer.html?model=qwen40_v2", flush=True)
    print(
        f"Perfetto trace endpoint: {trace_count} allowlisted files under "
        + ", ".join(str(root.resolve()) for root in trace_roots),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
