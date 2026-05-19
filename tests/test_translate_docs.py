import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import translate_docs as td


def test_dest_path_for():
    p = Path("src/content/docs/cards/creating-cards.mdx")
    dst = td.dest_path_for(p)
    assert dst.name == "creating-cards.pt-br.mdx"


def test_read_and_render_frontmatter(tmp_path):
    f = tmp_path / "sample.mdx"
    content = """---\ntitle: Test\n---\n\nHello world"""
    f.write_text(content, encoding="utf-8")
    front, body = td.read_frontmatter_and_body(f)
    assert "title: Test" in front
    assert "Hello world" in body
    full = td.render_with_frontmatter(front, body)
    assert full.startswith("---")


def test_write_translation_skip_and_force(tmp_path):
    src = tmp_path / "doc.mdx"
    src.write_text("Hello", encoding="utf-8")
    dst = td.dest_path_for(src)
    # create existing dst
    dst.write_text("Existing", encoding="utf-8")
    ok, path = td.write_translation(src, "Olá", None, dry_run=False, force=False)
    assert ok is False
    assert path == dst
    # force overwrite
    ok2, path2 = td.write_translation(src, "Olá", None, dry_run=False, force=True)
    assert ok2 is True
    assert path2.exists()


def test_process_file_monkeypatched(tmp_path, monkeypatch):
    src = tmp_path / "doc.mdx"
    src.write_text("Hello world", encoding="utf-8")

    def fake_translate(text, model="gemma"):
        return "Olá mundo"

    monkeypatch.setattr(td, "translate_text", fake_translate)
    ok, dst = td.process_file(src, model="gemma", dry_run=False, force=True)
    assert ok is True
    assert dst.exists()
    txt = dst.read_text(encoding="utf-8")
    assert "Olá mundo" in txt
