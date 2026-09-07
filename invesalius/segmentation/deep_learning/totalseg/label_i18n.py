# --------------------------------------------------------------------------
# Software:     InVesalius - Software de Reconstrucao 3D de Imagens Medicas
# Copyright:    (C) 2001  Centro de Pesquisas Renato Archer
# Homepage:     http://www.softwarepublico.gov.br
# Contact:      invesalius@cti.gov.br
# License:      GNU - GPL 2 (LICENSE.txt/LICENCA.txt)
# --------------------------------------------------------------------------
# Structure keys come from external TotalSegmentator sidecars, so pygettext
# can't extract them. Anatomical vocabulary is translated via JSON tables here
# instead of the project-wide .po files. Lookup: session locale -> English ->
# auto-formatted key.

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

TRANSLATIONS_DIR = Path(__file__).parent / "translations"
_DEFAULT_LOCALE = "en"


def _display_from_key(name: str) -> str:
    return " ".join(w.capitalize() for w in name.split("_"))


@lru_cache(maxsize=16)
def _load_locale(locale: str) -> dict:
    path = TRANSLATIONS_DIR / f"structures.{locale}.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("structures.%s.json is not a JSON object; ignoring", locale)
            return {}
        return data
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read structures.%s.json: %s", locale, e)
        return {}


def _current_session_locale() -> str:
    try:
        from invesalius.session import Session

        lang = Session().GetConfig("language")
        if isinstance(lang, str) and lang:
            return lang
    except Exception:  # noqa: BLE001
        pass
    return _DEFAULT_LOCALE


def translate_structure(name: str, locale: str | None = None) -> str:
    resolved_locale = locale or _current_session_locale()

    for candidate in (resolved_locale, _DEFAULT_LOCALE):
        translations = _load_locale(candidate)
        if name in translations:
            translated = translations[name]
            if isinstance(translated, str) and translated.strip():
                return translated

    return _display_from_key(name)


def available_locales() -> list[str]:
    if not TRANSLATIONS_DIR.is_dir():
        return []
    prefix = "structures."
    suffix = ".json"
    out = []
    for p in TRANSLATIONS_DIR.iterdir():
        n = p.name
        if n.startswith(prefix) and n.endswith(suffix):
            out.append(n[len(prefix) : -len(suffix)])
    return sorted(out)


def reset_cache() -> None:
    _load_locale.cache_clear()
