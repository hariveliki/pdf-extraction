"""The batch command: extract a folder, then move each processed PDF to input/."""

from __future__ import annotations

import shutil

import pytest

from pdfx.cli import main


@pytest.fixture
def cfg_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("pipeline:\n  native_backend: pymupdf\n")
    return path


def _batch(cfg_file, inbox, dest, out):
    return main(["-c", str(cfg_file), "batch", str(inbox),
                 "--move-to", str(dest), "-o", str(out)])


def test_batch_extracts_then_moves_each_pdf(native_pdf, cfg_file, tmp_path):
    inbox, dest, out = tmp_path / "inbox", tmp_path / "input", tmp_path / "out"
    (inbox / "sub").mkdir(parents=True)
    shutil.copy(native_pdf, inbox / "a.pdf")
    shutil.copy(native_pdf, inbox / "sub" / "b.pdf")

    assert _batch(cfg_file, inbox, dest, out) == 0

    assert (dest / "a.pdf").exists() and (dest / "sub" / "b.pdf").exists()
    assert not list(inbox.rglob("*.pdf")), "processed PDFs must leave the inbox"
    assert (out / "a.md").exists()


def test_batch_keeps_pdf_when_destination_name_is_taken(native_pdf, cfg_file, tmp_path):
    inbox, dest, out = tmp_path / "inbox", tmp_path / "input", tmp_path / "out"
    inbox.mkdir()
    dest.mkdir()
    shutil.copy(native_pdf, inbox / "a.pdf")
    (dest / "a.pdf").write_bytes(b"existing")

    _batch(cfg_file, inbox, dest, out)

    assert (inbox / "a.pdf").exists(), "never overwrite a file already in input/"
    assert (dest / "a.pdf").read_bytes() == b"existing"


def test_batch_rejects_a_single_file(native_pdf, cfg_file, tmp_path):
    assert _batch(cfg_file, native_pdf, tmp_path / "input", tmp_path / "out") == 3
    assert native_pdf.exists()


def test_extract_leaves_files_in_place(native_pdf, cfg_file, tmp_path):
    assert main(["-c", str(cfg_file), "extract", str(native_pdf),
                 "-o", str(tmp_path / "out")]) == 0
    assert native_pdf.exists()
