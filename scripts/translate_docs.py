import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

try:
    import yaml
except Exception:
    yaml = None


FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


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


def translate_text(text: str, model: str = "gemma") -> str:
    # Default implementation calls ollama; tests should monkeypatch this.
    try:
        proc = subprocess.run(
            [
                "ollama",
                "run",
                model,
                "--prompt",
                text,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError(
            "'ollama' CLI not found. Install ollama or monkeypatch translate_text in tests."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ollama call failed: {e.stderr}")


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


def process_file(
    path: Path, model: str = "gemma", dry_run: bool = False, force: bool = False
):
    front, body = read_frontmatter_and_body(path)
    prompt = f"Translate the following markdown text from English to Brazilian Portuguese, preserving code blocks, frontmatter keys, links and formatting.\n\n{body}"
    translated = translate_text(prompt, model=model)
    ok, dst = write_translation(path, translated, front, dry_run=dry_run, force=force)
    return ok, dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="src/content/docs", help="Root docs directory")
    ap.add_argument("--file", help="Single file to translate")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--force", action="store_true", help="Allow overwrite of existing translation"
    )
    ap.add_argument("--model", default="gemma")
    args = ap.parse_args()

    root = Path(args.dir)
    if args.file:
        p = Path(args.file)
        ok, dst = process_file(
            p, model=args.model, dry_run=args.dry_run, force=args.force
        )
        print(f"{p} -> {dst} : {'written' if ok else 'skipped (exists)'}")
        return

    for src in find_source_files(root):
        dst = dest_path_for(src)
        if dst.exists() and not args.force:
            print(f"skip existing: {dst}")
            continue
        try:
            ok, written = process_file(
                src, model=args.model, dry_run=args.dry_run, force=args.force
            )
            print(f"{src} -> {written} : {'written' if ok else 'skipped'}")
        except Exception as e:
            print(f"error processing {src}: {e}")


if __name__ == "__main__":
    main()
