import json
from pathlib import Path

from invesalius.segmentation.deep_learning.totalseg import label_i18n


def _redirect_translations(monkeypatch, tmp_path: Path) -> Path:
    """Point label_i18n at a throwaway translations directory and clear its cache."""
    monkeypatch.setattr(label_i18n, "TRANSLATIONS_DIR", tmp_path)
    label_i18n.reset_cache()
    return tmp_path


def _write(dir_: Path, locale: str, data: dict) -> None:
    (dir_ / f"structures.{locale}.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


def test_translate_returns_locale_value_when_present(monkeypatch, tmp_path):
    _redirect_translations(monkeypatch, tmp_path)
    _write(tmp_path, "en", {"kidney_right": "Kidney Right"})
    _write(tmp_path, "pt_BR", {"kidney_right": "Rim Direito"})

    assert label_i18n.translate_structure("kidney_right", locale="pt_BR") == "Rim Direito"


def test_translate_falls_back_to_english_when_locale_missing_key(monkeypatch, tmp_path):
    _redirect_translations(monkeypatch, tmp_path)
    _write(tmp_path, "en", {"kidney_right": "Kidney Right"})
    _write(tmp_path, "pt_BR", {})

    assert label_i18n.translate_structure("kidney_right", locale="pt_BR") == "Kidney Right"


def test_translate_falls_back_to_english_when_locale_file_absent(monkeypatch, tmp_path):
    _redirect_translations(monkeypatch, tmp_path)
    _write(tmp_path, "en", {"kidney_right": "Kidney Right"})

    assert label_i18n.translate_structure("kidney_right", locale="xx_YY") == "Kidney Right"


def test_translate_auto_formats_when_no_translation_exists(monkeypatch, tmp_path):
    _redirect_translations(monkeypatch, tmp_path)
    _write(tmp_path, "en", {})

    assert label_i18n.translate_structure("vertebrae_L1", locale="en") == "Vertebrae L1"
    assert label_i18n.translate_structure("kidney_right", locale="en") == "Kidney Right"


def test_translate_handles_missing_translations_dir(monkeypatch, tmp_path):
    # tmp_path exists but contains no files -> should still fall through cleanly.
    _redirect_translations(monkeypatch, tmp_path)

    assert label_i18n.translate_structure("liver", locale="en") == "Liver"


def test_translate_ignores_empty_or_whitespace_translation(monkeypatch, tmp_path):
    _redirect_translations(monkeypatch, tmp_path)
    _write(tmp_path, "en", {"liver": "Liver"})
    _write(tmp_path, "pt_BR", {"liver": "   "})

    # empty pt_BR value must NOT satisfy the lookup; fall through to English.
    assert label_i18n.translate_structure("liver", locale="pt_BR") == "Liver"


def test_translate_ignores_non_string_translation_value(monkeypatch, tmp_path):
    _redirect_translations(monkeypatch, tmp_path)
    _write(tmp_path, "en", {"liver": "Liver"})
    _write(tmp_path, "pt_BR", {"liver": 42})

    assert label_i18n.translate_structure("liver", locale="pt_BR") == "Liver"


def test_load_locale_handles_malformed_json(monkeypatch, tmp_path):
    _redirect_translations(monkeypatch, tmp_path)
    (tmp_path / "structures.en.json").write_text("{not: valid json", encoding="utf-8")

    # Corrupt file must not raise; falls through to auto-formatting.
    assert label_i18n.translate_structure("liver", locale="en") == "Liver"


def test_load_locale_handles_non_dict_json(monkeypatch, tmp_path):
    _redirect_translations(monkeypatch, tmp_path)
    (tmp_path / "structures.en.json").write_text('["not", "a", "dict"]', encoding="utf-8")

    assert label_i18n.translate_structure("liver", locale="en") == "Liver"


def test_available_locales_lists_files_present(monkeypatch, tmp_path):
    _redirect_translations(monkeypatch, tmp_path)
    _write(tmp_path, "en", {})
    _write(tmp_path, "pt_BR", {})
    _write(tmp_path, "de_DE", {})
    # Non-matching file that must be ignored.
    (tmp_path / "readme.txt").write_text("not a translation", encoding="utf-8")

    assert label_i18n.available_locales() == ["de_DE", "en", "pt_BR"]


def test_available_locales_empty_when_dir_missing(monkeypatch, tmp_path):
    missing = tmp_path / "does_not_exist"
    monkeypatch.setattr(label_i18n, "TRANSLATIONS_DIR", missing)
    label_i18n.reset_cache()

    assert label_i18n.available_locales() == []


def test_shipped_english_file_covers_expected_structures():
    # Real file check: the shipped structures.en.json should be a non-empty dict
    # with sensible entries. Guards accidental deletion or corruption.
    label_i18n.reset_cache()
    en = label_i18n._load_locale("en")
    assert isinstance(en, dict)
    assert len(en) > 100  # We ship ~121 structure names
    # A few canonical checks.
    assert en.get("kidney_right") == "Kidney Right"
    assert en.get("liver") == "Liver"
    assert en.get("vertebrae_L1") == "Vertebrae L1"
