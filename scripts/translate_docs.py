"""
translate_docs.py — Translate Huly/TraceX docs from English to Brazilian Portuguese
using the Ollama REST API (stdlib only, no subprocess).

Usage examples:
    python scripts/translate_docs.py --file src/content/docs/cards/creating-cards.mdx
    python scripts/translate_docs.py --dir src/content/docs --force
    python scripts/translate_docs.py --dir src/content/docs --concurrency 8 --glossary scripts/glossary.json
"""

import argparse
import concurrent.futures
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import yaml
except Exception:
    yaml = None


# ---------------------------------------------------------------------------
# Regex to split YAML front-matter from the document body.
# ---------------------------------------------------------------------------
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


# ===========================================================================
# Unchanged helper functions (preserved from original implementation)
# ===========================================================================

def read_frontmatter_and_body(path: Path) -> Tuple[Optional[str], str]:
    text = path.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    if m:
        return m.group(1), m.group(2)
    return None, text


def render_with_frontmatter(front: Optional[str], body: str) -> str:
    if front:
        return f"---\n{front}\n---\n\n{body}"
    return body


def dest_path_for(source: Path) -> Path:
    # e.g. creating-cards.mdx -> creating-cards.pt-br.mdx
    if source.suffix == ".mdx":
        return source.with_name(source.stem + ".pt-br" + source.suffix)
    return source.with_suffix(source.suffix + ".pt-br")


def find_source_files(root: Path):
    for p in root.rglob("*.mdx"):
        yield p


def write_translation(
    source: Path,
    translated: str,
    front: Optional[str],
    dry_run: bool = False,
    force: bool = False,
) -> Tuple[bool, Path]:
    dst = dest_path_for(source)
    if dst.exists() and not force:
        return False, dst
    content = render_with_frontmatter(front, translated)
    if dry_run:
        return True, dst
    dst.write_text(content, encoding="utf-8")
    return True, dst


# ===========================================================================
# New / rewritten functions
# ===========================================================================


def load_glossary(path: Optional[Path] = None) -> dict:
    """Load a JSON glossary mapping English terms to Brazilian Portuguese.

    If *path* is None, try ``scripts/glossary.json`` relative to this script.
    Returns an empty dict if the file does not exist — never raises on a missing file.
    """
    if path is None:
        path = Path(__file__).parent / "glossary.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def build_prompt(chunk: str, glossary: dict) -> str:
    """Build the full translation prompt for *chunk* with an optional glossary table."""
    lines = [
        "Translate the following markdown from English to Brazilian Portuguese (pt-br).",
        "- Preserve code blocks, links, YAML keys, URLs, and all formatting.",
        "- Use formal Brazilian Portuguese. Use 'você', not 'tu'. Avoid Portugal expressions.",
        "- Do NOT translate: code identifiers, URLs, YAML frontmatter keys, proper nouns.",
    ]
    if glossary:
        lines.append("")
        lines.append("Glossary (use these translations consistently):")
        lines.append("| English | Português |")
        lines.append("|---------|----------|")
        for en, pt in glossary.items():
            lines.append(f"| {en} | {pt} |")
    lines.append("")
    lines.append("Content to translate:")
    lines.append(chunk)
    return "\n".join(lines)


def translate_text(
    text: str,
    model: str = "translategemma:4b",
    ollama_url: str = "http://localhost:11434",
) -> str:
    """POST *text* as a prompt to the Ollama REST API and return the response.

    Raises RuntimeError on network failure or an unexpected response shape.
    """
    url = f"{ollama_url}/api/generate"
    payload = json.dumps({"model": model, "prompt": text, "stream": False}).encode(
        "utf-8"
    )
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach ollama at {ollama_url}: {e}") from e

    try:
        data = json.loads(raw)
        return data["response"].strip()
    except (KeyError, json.JSONDecodeError):
        raise RuntimeError(f"Unexpected ollama response: {raw}")


def split_into_chunks(body: str, chunk_words: int = 400) -> List[str]:
    """Split *body* into translatable chunks.

    Rules:
    - Split at markdown heading lines (``# ...`` through ``###### ...``).
    - Each heading line opens a new chunk.
    - Chunks exceeding *chunk_words* words are further split at double-newline
      paragraph boundaries.
    - Never split inside a fenced code block (triple backticks).
    - ``"".join(split_into_chunks(body))`` is guaranteed to equal *body*.
    """
    fence_re = re.compile(r"^```", re.MULTILINE)
    heading_re = re.compile(r"^#{1,6} ", re.MULTILINE)

    def _inside_fence(text: str, pos: int) -> bool:
        """True if character at *pos* falls inside an open code fence."""
        return len(fence_re.findall(text[:pos])) % 2 == 1

    # ------------------------------------------------------------------
    # Phase 1: split at heading boundaries (skipping fenced headings).
    # ------------------------------------------------------------------
    split_positions: List[int] = [0]
    for m in heading_re.finditer(body):
        if m.start() == 0:
            continue  # position 0 is already the first split point
        if not _inside_fence(body, m.start()):
            split_positions.append(m.start())

    raw_chunks: List[str] = []
    for i, start in enumerate(split_positions):
        end = split_positions[i + 1] if i + 1 < len(split_positions) else len(body)
        raw_chunks.append(body[start:end])

    # ------------------------------------------------------------------
    # Phase 2: further split oversized chunks at paragraph boundaries,
    #          never splitting inside a fenced code block.
    # ------------------------------------------------------------------

    def _split_by_paragraphs(chunk: str) -> List[str]:
        """Split *chunk* on double-newlines while respecting code fences.

        ``"".join(result)`` is guaranteed to equal *chunk*.
        """
        para_sep = "\n\n"
        paragraphs = chunk.split(para_sep)
        parts: List[str] = []
        current = ""
        in_fence = False

        for idx, para in enumerate(paragraphs):
            sep = para_sep if idx < len(paragraphs) - 1 else ""
            fence_count = para.count("```")

            if in_fence:
                # Inside a fence — must not break here; keep accumulating.
                current += para + sep
                if fence_count % 2 == 1:
                    in_fence = False  # closing fence found
            else:
                current += para + sep
                if fence_count % 2 == 1:
                    in_fence = True  # opening fence found; defer split
                else:
                    parts.append(current)
                    current = ""

        if current:
            parts.append(current)

        return parts if parts else [chunk]

    final_chunks: List[str] = []
    for chunk in raw_chunks:
        if len(chunk.split()) > chunk_words:
            final_chunks.extend(_split_by_paragraphs(chunk))
        else:
            final_chunks.append(chunk)

    # Invariant: reassembly must reproduce the original body exactly.
    assert (
        "".join(final_chunks) == body
    ), "split_into_chunks: reassembly mismatch — this is a bug"

    return final_chunks


def translate_chunks_parallel(
    chunks: List[str],
    model: str,
    concurrency: int,
    glossary: dict,
    ollama_url: str,
) -> List[str]:
    """Translate *chunks* concurrently, preserving input order.

    Uses a ``ThreadPoolExecutor`` with *concurrency* workers.
    Each chunk is translated via :func:`translate_text` after building its
    prompt with :func:`build_prompt`.
    """

    def _translate_one(chunk: str) -> str:
        prompt = build_prompt(chunk, glossary)
        return translate_text(prompt, model=model, ollama_url=ollama_url)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(_translate_one, chunks))
    return results


# ===========================================================================
# Orchestration
# ===========================================================================


def process_file(
    path: Path,
    model: str = "translategemma:4b",
    dry_run: bool = False,
    force: bool = False,
    concurrency: int = 4,
    chunk_words: int = 400,
    glossary: Optional[dict] = None,
    ollama_url: str = "http://localhost:11434",
) -> Tuple[bool, Path]:
    if glossary is None:
        glossary = {}

    front, body = read_frontmatter_and_body(path)

    if len(body.split()) > chunk_words:
        chunks = split_into_chunks(body, chunk_words=chunk_words)
        translated_parts = translate_chunks_parallel(
            chunks,
            model=model,
            concurrency=concurrency,
            glossary=glossary,
            ollama_url=ollama_url,
        )
        translated = "".join(translated_parts)
    else:
        translated = translate_text(
            build_prompt(body, glossary), model=model, ollama_url=ollama_url
        )

    assert translated, "Translation returned empty result"

    ok, dst = write_translation(path, translated, front, dry_run=dry_run, force=force)
    return ok, dst


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Translate Huly/TraceX MDX docs from English to Brazilian Portuguese."
    )
    ap.add_argument("--dir", default="src/content/docs", help="Root docs directory")
    ap.add_argument("--file", help="Single file to translate")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--force", action="store_true", help="Overwrite existing translations"
    )
    ap.add_argument("--model", default="translategemma:4b", help="Ollama model name")
    ap.add_argument(
        "--concurrency", type=int, default=4, help="Parallel worker threads"
    )
    ap.add_argument("--chunk-words", type=int, default=400, help="Max words per chunk")
    ap.add_argument("--glossary", default=None, help="Path to JSON glossary file")
    ap.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base URL of the Ollama server",
    )
    args = ap.parse_args()

    glossary = load_glossary(Path(args.glossary) if args.glossary else None)
    if glossary:
        print(f"Loaded glossary with {len(glossary)} terms.")

    common = dict(
        model=args.model,
        dry_run=args.dry_run,
        force=args.force,
        concurrency=args.concurrency,
        chunk_words=args.chunk_words,
        glossary=glossary,
        ollama_url=args.ollama_url,
    )

    if args.file:
        p = Path(args.file)
        ok, dst = process_file(p, **common)
        print(f"{p} -> {dst} : {'written' if ok else 'skipped (exists)'}")
        return

    root = Path(args.dir)
    for src in find_source_files(root):
        dst = dest_path_for(src)
        if dst.exists() and not args.force:
            print(f"skip existing: {dst}")
            continue
        try:
            ok, written = process_file(src, **common)
            print(f"{src} -> {written} : {'written' if ok else 'skipped'}")
        except Exception as e:
            print(f"error processing {src}: {e}")


if __name__ == "__main__":
    main()
