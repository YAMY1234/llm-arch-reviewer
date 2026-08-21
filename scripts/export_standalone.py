#!/usr/bin/env python3
"""Export one V2 viewer model as a self-contained offline HTML file."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path
import re
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ELK_URL = "https://unpkg.com/elkjs@0.9.3/lib/elk.bundled.js"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="built docs model id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docs-root", type=Path, default=REPO_ROOT / "docs")
    parser.add_argument("--viewer", type=Path, default=REPO_ROOT / "docs" / "viewer.html")
    parser.add_argument("--elk-js", type=Path, help="optional local elk.bundled.js")
    parser.add_argument("--elk-url", default=DEFAULT_ELK_URL)
    return parser.parse_args()


def b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def load_elk(args: argparse.Namespace) -> str:
    if args.elk_js:
        return args.elk_js.read_text()
    with urlopen(args.elk_url, timeout=30) as response:  # noqa: S310 - pinned URL
        return response.read().decode("utf-8")


def embedded_bootstrap(
    *, model_id: str, arch_gzip: bytes, timelines: dict[str, bytes]
) -> str:
    timeline_payload = {name: b64(payload) for name, payload in timelines.items()}
    return f"""<script>
window.__LLM_ARCH_EMBEDDED_MODEL_ID__ = {json.dumps(model_id)};
window.__LLM_ARCH_STANDALONE__ = true;
(function() {{
  const embeddedArchGzip = {json.dumps(b64(arch_gzip))};
  const embeddedTimelines = {json.dumps(timeline_payload, separators=(',', ':'))};
  const nativeFetch = window.fetch.bind(window);
  function decodeBase64(value) {{
    const binary = atob(value);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }}
  window.fetch = async function(input, init) {{
    const raw = typeof input === 'string' ? input : input.url;
    const url = new URL(raw, location.href);
    const pathname = decodeURIComponent(url.pathname);
    if (pathname.endsWith('/{model_id}/arch_data.json') || raw.startsWith('{model_id}/arch_data.json')) {{
      if (typeof DecompressionStream === 'undefined') {{
        throw new Error('This standalone viewer requires a browser with gzip DecompressionStream support.');
      }}
      const compressed = decodeBase64(embeddedArchGzip);
      const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'));
      return new Response(stream, {{headers: {{'Content-Type': 'application/json'}}}});
    }}
    const timelineName = pathname.split('/').pop();
    if (timelineName && Object.prototype.hasOwnProperty.call(embeddedTimelines, timelineName)) {{
      return new Response(decodeBase64(embeddedTimelines[timelineName]), {{
        headers: {{'Content-Type': 'application/gzip'}}
      }});
    }}
    return nativeFetch(input, init);
  }};
}})();
</script>"""


def main() -> int:
    args = parse_args()
    model_root = args.docs_root / args.model
    arch_path = model_root / "arch_data.json"
    bundle = json.loads(arch_path.read_text())
    compact_arch = json.dumps(
        bundle, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    arch_gzip = gzip.compress(compact_arch, compresslevel=9, mtime=0)

    timeline_paths: dict[str, Path] = {}
    for profile in bundle.get("profiles", {}).values():
        timeline = (profile.get("meta") or {}).get("timeline")
        if not timeline:
            continue
        relative = Path(str(timeline["url"]))
        source = model_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        timeline_paths[source.name] = source
    timelines = {name: path.read_bytes() for name, path in sorted(timeline_paths.items())}

    viewer = args.viewer.read_text()
    elk = load_elk(args).replace("</script>", "<\\/script>")
    viewer, count = re.subn(
        r'<script\s+src="https://unpkg\.com/elkjs@0\.9\.3/lib/elk\.bundled\.js"></script>',
        lambda _: f"<script>\n{elk}\n</script>",
        viewer,
        count=1,
    )
    if count != 1:
        raise ValueError("viewer ELK script tag was not found")

    bootstrap = embedded_bootstrap(
        model_id=args.model, arch_gzip=arch_gzip, timelines=timelines
    )
    marker = '<script>\n"use strict";'
    if marker not in viewer:
        raise ValueError("viewer application script marker was not found")
    viewer = viewer.replace(marker, f"{bootstrap}\n{marker}", 1)
    viewer = viewer.replace(
        "<title>llm-arch-reviewer</title>",
        f"<title>{args.model} · standalone llm-arch-reviewer</title>",
        1,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(viewer)
    print(f"wrote {args.output}")
    print(
        f"standalone bundle: profiles={len(bundle.get('profiles', {}))} "
        f"timelines={len(timelines)} size_mib={args.output.stat().st_size / 1024 / 1024:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
