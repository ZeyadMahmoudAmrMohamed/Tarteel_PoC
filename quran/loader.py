"""
quran/loader.py
---------------
Loads the Quran corpus (with full tashkeel) from a local JSON file
or downloads it automatically from a reliable open-source API.

Data format expected (and what we download):
  {
    "1": {                          # surah number
      "1": {                        # ayah number
        "text": "بِسْمِ اللَّهِ ..."   # full text with diacritics
      },
      ...
    },
    ...
  }

We use the Quran API v2 (api.alquran.cloud) which provides:
  - Full Arabic text with diacritics (hafs 'an 'asim riwaya)
  - All 114 surahs, 6236 ayahs

The downloaded data is cached locally at quran/data/quran.json so
subsequent runs are instant and offline-capable.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error
import os
os.makedirs("/kaggle/working/quran/data", exist_ok=True)
logger = logging.getLogger(__name__)

# Path relative to this file
_DATA_DIR = Path(__file__).parent / "data"
_DEFAULT_CACHE = _DATA_DIR / "quran.json"

# Quran API — returns all 6236 ayahs with tashkeel in one call
_API_URL = "https://api.alquran.cloud/v1/quran/quran-uthmani"


class QuranLoader:
    """
    Loads the Quran corpus into memory.

    Parameters
    ----------
    cache_path : Path to local JSON cache. Downloads if missing.
    auto_download : If True (default), download on first run.
    """

    def __init__(
        self,
        cache_path: str | Path | None = None,
        auto_download: bool = True,
    ) -> None:
        self.cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE
        self.auto_download = auto_download
        self._data: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> dict:
        """
        Load Quran data. Returns the full nested dict:
            { surah_num_str: { ayah_num_str: { "text": str } } }

        Downloads and caches automatically if local file is missing.
        """
        if self._data is not None:
            return self._data

        if self.cache_path.exists():
            logger.info(f"[QuranLoader] Loading from cache: {self.cache_path}")
            self._data = self._load_from_file(self.cache_path)
        elif self.auto_download:
            logger.info("[QuranLoader] Cache not found — downloading from API …")
            self._data = self._download_and_cache()
        else:
            raise FileNotFoundError(
                f"Quran data not found at {self.cache_path}. "
                "Set auto_download=True or provide the file manually."
            )

        total = sum(len(v) for v in self._data.values())
        logger.info(
            f"[QuranLoader] Loaded {len(self._data)} surahs, {total} ayahs"
        )
        return self._data

    def get_verse(self, surah: int, ayah: int) -> Optional[str]:
        """Return text of a specific verse (with diacritics), or None."""
        data = self.load()
        try:
            return data[str(surah)][str(ayah)]["text"]
        except KeyError:
            return None

    def get_surah(self, surah: int) -> dict:
        """Return all ayahs of a surah as { ayah_num_str: {text:...} }."""
        data = self.load()
        result = data.get(str(surah))
        if result is None:
            raise ValueError(f"Surah {surah} not found.")
        return result

    def iter_verses(self):
        """Yield (surah_int, ayah_int, text_with_diac) for every verse."""
        data = self.load()
        for s_str, ayahs in data.items():
            for a_str, content in ayahs.items():
                yield int(s_str), int(a_str), content["text"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_from_file(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _download_and_cache(self) -> dict:
        """
        Downloads full Quran from alquran.cloud API and saves to cache.
        """
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                _API_URL,
                headers={"User-Agent": "tarteel-poc/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Failed to download Quran data: {e}\n"
                "Check your internet connection, or place quran.json manually at:\n"
                f"  {self.cache_path}"
            ) from e

        logger.info(f"[QuranLoader] API response received in {time.perf_counter()-t0:.1f}s")

        # Parse API response into our nested format
        data = self._parse_api_response(raw)

        # Ensure data dir exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"[QuranLoader] Saved cache to {self.cache_path}")
        return data

    @staticmethod
    def _parse_api_response(raw: dict) -> dict:
        """
        alquran.cloud /v1/quran response structure:
          raw["data"]["surahs"] = [
            {
              "number": 1,
              "ayahs": [
                { "numberInSurah": 1, "text": "..." },
                ...
              ]
            },
            ...
          ]
        """
        data: dict = {}
        surahs = raw.get("data", {}).get("surahs", [])
        for surah_obj in surahs:
            s_num = str(surah_obj["number"])
            data[s_num] = {}
            for ayah_obj in surah_obj.get("ayahs", []):
                a_num = str(ayah_obj["numberInSurah"])
                text = ayah_obj["text"]
                # Remove the small circle (U+06DD, End of Ayah) if present
                text = text.replace("\u06dd", "").strip()
                data[s_num][a_num] = {"text": text}
        return data