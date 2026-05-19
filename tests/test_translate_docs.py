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

    def fake_translate(
        text, model="translategemma:4b", ollama_url="http://localhost:11434"
    ):
        return "Olá mundo"

    monkeypatch.setattr(td, "translate_text", fake_translate)
    ok, dst = td.process_file(src, model="translategemma:4b", dry_run=False, force=True)
    assert ok is True
    assert dst.exists()
    txt = dst.read_text(encoding="utf-8")
    assert "Olá mundo" in txt


# ---------------------------------------------------------------------------
# New tests — chunking
# ---------------------------------------------------------------------------


def test_split_into_chunks_by_heading():
    body = (
        "## Introduction\n\nSome intro text.\n\n"
        "## Features\n\nFeature description.\n\n"
        "## Setup\n\nSetup instructions."
    )
    chunks = td.split_into_chunks(body, chunk_words=400)
    heading_chunks = [c for c in chunks if c.strip().startswith("#")]
    assert len(heading_chunks) >= 3
    assert "".join(chunks) == body


def test_split_into_chunks_no_split_in_code_block():
    code_lines = ["    line " + str(i) for i in range(200)]
    code_block = "```python\n" + "\n".join(code_lines) + "\n```"
    body = "## My Section\n\n" + code_block + "\n\nEnd of section."
    chunks = td.split_into_chunks(body, chunk_words=50)
    for chunk in chunks:
        opens = chunk.count("```")
        assert opens % 2 == 0, f"Unbalanced fences in chunk: {chunk[:80]!r}"
    assert "".join(chunks) == body


# ---------------------------------------------------------------------------
# New tests — build_prompt & glossary
# ---------------------------------------------------------------------------


def test_build_prompt_includes_glossary():
    glossary = {"issue": "issue", "label": "etiqueta"}
    prompt = td.build_prompt("Hello world", glossary)
    assert "issue" in prompt
    assert "etiqueta" in prompt
    assert "Hello world" in prompt
    assert "Brazilian Portuguese" in prompt


def test_build_prompt_no_glossary_table():
    prompt = td.build_prompt("Hello", {})
    assert "Glossary" not in prompt
    assert "Hello" in prompt


def test_load_glossary_returns_dict(tmp_path):
    import json

    data = {"feature": "funcionalidade", "build": "build"}
    f = tmp_path / "glossary.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    result = td.load_glossary(f)
    assert result == data


def test_load_glossary_missing_returns_empty():
    result = td.load_glossary(Path("/nonexistent/path/glossary.json"))
    assert result == {}


# ---------------------------------------------------------------------------
# New tests — translate_chunks_parallel
# ---------------------------------------------------------------------------


def test_translate_chunks_parallel_monkeypatched(monkeypatch):
    call_log = []

    def fake_translate(
        text, model="translategemma:4b", ollama_url="http://localhost:11434"
    ):
        call_log.append(text)
        return "translated"

    monkeypatch.setattr(td, "translate_text", fake_translate)
    results = td.translate_chunks_parallel(
        ["chunk a", "chunk b", "chunk c"],
        model="translategemma:4b",
        concurrency=2,
        glossary={},
        ollama_url="http://localhost:11434",
    )
    assert results == ["translated", "translated", "translated"]
    assert len(call_log) == 3


# ---------------------------------------------------------------------------
# New tests — process_file with chunking path
# ---------------------------------------------------------------------------


def test_process_file_uses_chunks(tmp_path, monkeypatch):
    paragraph = "This is a sentence with several words. " * 20
    body = "\n\n".join([paragraph] * 5)
    src = tmp_path / "long_doc.mdx"
    src.write_text(body, encoding="utf-8")

    translated_chunks = []

    def fake_translate(
        text, model="translategemma:4b", ollama_url="http://localhost:11434"
    ):
        translated_chunks.append(text)
        return "Texto traduzido."

    monkeypatch.setattr(td, "translate_text", fake_translate)
    ok, dst = td.process_file(src, chunk_words=100, force=True)
    assert ok is True
    assert dst.exists()
    content = dst.read_text(encoding="utf-8")
    assert "Texto traduzido." in content
    assert len(translated_chunks) > 1


# ---------------------------------------------------------------------------
# New tests — network error handling
# ---------------------------------------------------------------------------


def test_translate_text_url_error(monkeypatch):
    import pytest
    import urllib.error

    def mock_urlopen(req, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(td.urllib.request, "urlopen", mock_urlopen)
    with pytest.raises(RuntimeError, match="Cannot reach ollama"):
        td.translate_text(
            "hello", model="translategemma:4b", ollama_url="http://localhost:11434"
        )
