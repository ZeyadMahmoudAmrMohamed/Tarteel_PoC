"""
core/alignment.py
-----------------
Aligns the raw Whisper transcription output against the ground-truth
Quranic text (with full tashkeel / diacritics).

The challenge
-------------
  Whisper outputs text WITHOUT diacritics (harakaat).
  The ground truth has full diacritics (fatha, kasra, damma, shadda, etc.)
  We need to:
    1. Strip diacritics from the ground truth for comparison.
    2. Find the best-matching verse/segment.
    3. Return the ground truth text with its original diacritics, annotated
       with which characters were correctly recognised and which were errors.

Approach
--------
  - Normalise both texts (remove diacritics, normalise alef forms, etc.)
  - Use sequence alignment (Levenshtein / rapidfuzz) at the word level
    to find the best matching position in the corpus.
  - Map each word back to its diacritic-annotated ground truth token.

Diacritic Unicode codepoints (for reference):
  U+064B  ARABIC FATHATAN
  U+064C  ARABIC DAMMATAN
  U+064D  ARABIC KASRATAN
  U+064E  ARABIC FATHA
  U+064F  ARABIC DAMMA
  U+0650  ARABIC KASRA
  U+0651  ARABIC SHADDA
  U+0652  ARABIC SUKUN
  U+0653  ARABIC MADDAH ABOVE
  U+0654  ARABIC HAMZA ABOVE
  U+0655  ARABIC HAMZA BELOW
  U+0670  ARABIC LETTER SUPERSCRIPT ALEF
"""

from __future__ import annotations

import re
import logging
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

# ── Arabic Unicode ranges ──────────────────────────────────────────────────────

DIACRITICS_RE = re.compile(
    r"[\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)

# Alef variants → plain alef
ALEF_VARIANTS_RE = re.compile(r"[\u0622\u0623\u0625\u0671]")

# Normalise wa (waw with hamza above)
WAW_RE = re.compile(r"\u0624")

# Normalise ya without dots (alef maqsura) and ya with dots
YA_RE = re.compile(r"\u0649")


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class WordAlignment:
    """Alignment result for a single word."""
    hyp_word: str          # what Whisper said (no diacritics)
    ref_word: str          # ground truth word (no diacritics)
    ref_word_with_diac: str  # ground truth with diacritics
    status: str            # "correct" | "substitution" | "insertion" | "deletion"
    similarity: float      # 0.0 – 1.0


@dataclass
class AlignmentResult:
    """Full alignment result for a transcription against a verse."""

    # Identification
    surah: int
    ayah: int

    # Texts
    hypothesis: str              # raw Whisper output (no diacritics)
    reference_plain: str         # ground truth stripped of diacritics
    reference_with_diacritics: str  # ground truth as-is

    # Alignment details
    word_alignments: list[WordAlignment]

    # Metrics
    wer: float                   # word error rate (0.0 – 1.0)
    cer: float                   # character error rate (0.0 – 1.0)
    match_score: float           # fuzzy match score used to pick this verse (0–100)

    # Convenience
    correctly_recited_words: list[str] = field(default_factory=list)
    error_words: list[str] = field(default_factory=list)


# ── Main aligner ───────────────────────────────────────────────────────────────

class QuranAligner:
    """
    Aligns Whisper transcription output with Quranic ground truth.

    Parameters
    ----------
    quran_data : dict loaded from quran/loader.py — structure:
        {
          "1": {           # surah number (str)
            "1": {         # ayah number (str)
              "text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
            },
            ...
          },
          ...
        }
    similarity_threshold : Minimum fuzzy match score to accept a verse candidate.
    """

    def __init__(self, quran_data: dict, similarity_threshold: float = 40.0) -> None:
        self.quran_data = quran_data
        self.similarity_threshold = similarity_threshold

        # Pre-build search index: flat list of (surah, ayah, plain_text, original_text)
        self._index = self._build_index()
        logger.info(f"[QuranAligner] Index built: {len(self._index)} verses")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(
        self,
        transcription: str,
        surah: Optional[int] = None,
        ayah: Optional[int] = None,
    ) -> AlignmentResult:
        """
        Align transcription against Quran.

        Parameters
        ----------
        transcription : Raw text from Whisper (no diacritics expected).
        surah         : If provided, restrict search to this surah.
        ayah          : If provided AND surah provided, align against exact verse.

        Returns
        -------
        AlignmentResult
        """
        hyp_normalised = self.normalise(transcription)

        # ── Case 1: exact verse specified ──────────────────────────────
        if surah is not None and ayah is not None:
            verse = self._get_verse(surah, ayah)
            if verse is None:
                raise ValueError(f"Verse {surah}:{ayah} not found in Quran data.")
            ref_plain = self.normalise(verse)
            score = fuzz.token_set_ratio(hyp_normalised, ref_plain)
            logger.info(
                f"[QuranAligner] Aligned against {surah}:{ayah} — score {score:.1f}"
            )
            return self._build_result(
                hyp=hyp_normalised,
                ref_plain=ref_plain,
                ref_diac=verse,
                surah=surah,
                ayah=ayah,
                match_score=float(score),
            )

        # ── Case 2: search within a surah ─────────────────────────────
        if surah is not None:
            candidates = [(s, a, p, d) for s, a, p, d in self._index if s == surah]
        else:
            candidates = self._index

        best = self._find_best_match(hyp_normalised, candidates)

        if best is None:
            raise RuntimeError(
                "No verse found above similarity threshold. "
                "Try lowering similarity_threshold or check the audio."
            )

        s, a, ref_plain, ref_diac, score = best
        logger.info(f"[QuranAligner] Best match: {s}:{a} — score {score:.1f}")

        return self._build_result(
            hyp=hyp_normalised,
            ref_plain=ref_plain,
            ref_diac=ref_diac,
            surah=s,
            ayah=a,
            match_score=float(score),
        )

    def normalise(self, text: str) -> str:
        """
        Normalise Arabic text for comparison:
          1. Remove diacritics (tashkeel).
          2. Normalise alef variants to plain alef (ا).
          3. Normalise waw with hamza (ؤ → و).
          4. Normalise alef maqsura (ى → ي).
          5. Remove tatweel (kashida, ـ).
          6. Collapse whitespace.
        """
        text = DIACRITICS_RE.sub("", text)
        text = ALEF_VARIANTS_RE.sub("\u0627", text)   # → ا
        text = WAW_RE.sub("\u0648", text)              # → و
        text = YA_RE.sub("\u064A", text)               # → ي
        text = text.replace("\u0640", "")              # remove tatweel
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_index(self) -> list[tuple]:
        """Build flat list: (surah_int, ayah_int, plain_text, original_text)"""
        index = []
        for surah_str, ayahs in self.quran_data.items():
            for ayah_str, content in ayahs.items():
                original = content["text"]
                plain = self.normalise(original)
                index.append((int(surah_str), int(ayah_str), plain, original))
        return index

    def _find_best_match(
        self, hypothesis: str, candidates: list[tuple]
    ) -> Optional[tuple]:
        """
        Find the verse in candidates that best matches the hypothesis.
        Returns (surah, ayah, plain, diac, score) or None.
        """
        best_score = -1.0
        best_entry = None

        # Build a mapping: plain_text → (surah, ayah, plain, diac)
        # We'll use rapidfuzz extractOne on plain texts
        plains = [c[2] for c in candidates]
        
        match = process.extractOne(
            hypothesis,
            plains,
            scorer=fuzz.token_set_ratio,
            score_cutoff=self.similarity_threshold,
        )

        if match is None:
            return None

        matched_plain, score, idx = match
        s, a, p, d = candidates[idx]
        return (s, a, p, d, score)

    def _get_verse(self, surah: int, ayah: int) -> Optional[str]:
        """Return original (with diacritics) text of a verse, or None."""
        try:
            return self.quran_data[str(surah)][str(ayah)]["text"]
        except KeyError:
            return None

    def _build_result(
        self,
        hyp: str,
        ref_plain: str,
        ref_diac: str,
        surah: int,
        ayah: int,
        match_score: float,
    ) -> AlignmentResult:
        """Run word-level alignment and compute metrics."""

        hyp_words = hyp.split()
        ref_words = ref_plain.split()
        ref_diac_words = ref_diac.split()

        # Pad ref_diac_words to same length as ref_words if needed
        # (they should match since diac→plain is 1-to-1 at word level)
        if len(ref_diac_words) != len(ref_words):
            # Fallback: rebuild from diac by splitting
            ref_diac_words = ref_diac.split()

        word_aligns = self._align_words(hyp_words, ref_words, ref_diac_words)

        wer = self._compute_wer(hyp_words, ref_words)
        cer = self._compute_cer(hyp, ref_plain)

        correct = [wa.ref_word_with_diac for wa in word_aligns if wa.status == "correct"]
        errors = [wa.hyp_word for wa in word_aligns if wa.status != "correct"]

        return AlignmentResult(
            surah=surah,
            ayah=ayah,
            hypothesis=hyp,
            reference_plain=ref_plain,
            reference_with_diacritics=ref_diac,
            word_alignments=word_aligns,
            wer=wer,
            cer=cer,
            match_score=match_score,
            correctly_recited_words=correct,
            error_words=errors,
        )

    def _align_words(
        self,
        hyp_words: list[str],
        ref_words: list[str],
        ref_diac_words: list[str],
    ) -> list[WordAlignment]:
        """
        Dynamic-programming word-level alignment (like standard WER alignment).
        Returns list of WordAlignment objects covering every reference word.
        """
        H, R = len(hyp_words), len(ref_words)

        # DP table: (insertions, deletions, substitutions)
        # Standard edit-distance DP
        dp = [[0] * (H + 1) for _ in range(R + 1)]
        for i in range(R + 1):
            dp[i][0] = i
        for j in range(H + 1):
            dp[0][j] = j

        for i in range(1, R + 1):
            for j in range(1, H + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # deletion (ref not said)
                        dp[i][j - 1],      # insertion (extra word said)
                        dp[i - 1][j - 1],  # substitution
                    )

        # Backtrack
        alignments: list[WordAlignment] = []
        i, j = R, H
        while i > 0 or j > 0:
            diac_word = ref_diac_words[i - 1] if i > 0 else ""
            ref_w = ref_words[i - 1] if i > 0 else ""
            hyp_w = hyp_words[j - 1] if j > 0 else ""

            if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
                status = "correct"
                sim = 1.0
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                status = "substitution"
                sim = fuzz.ratio(hyp_w, ref_w) / 100.0
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                status = "deletion"
                sim = 0.0
                hyp_w = ""
                i -= 1
            else:
                status = "insertion"
                sim = 0.0
                ref_w = ""
                diac_word = ""
                j -= 1

            alignments.append(
                WordAlignment(
                    hyp_word=hyp_w,
                    ref_word=ref_w,
                    ref_word_with_diac=diac_word,
                    status=status,
                    similarity=sim,
                )
            )

        alignments.reverse()
        return alignments

    @staticmethod
    def _compute_wer(hyp_words: list[str], ref_words: list[str]) -> float:
        """Standard Word Error Rate."""
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        H, R = len(hyp_words), len(ref_words)
        dp = list(range(H + 1))
        for i in range(1, R + 1):
            prev = dp[:]
            dp[0] = i
            for j in range(1, H + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    dp[j] = prev[j - 1]
                else:
                    dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
        return dp[H] / R

    @staticmethod
    def _compute_cer(hyp: str, ref: str) -> float:
        """Character Error Rate (on sequences without spaces)."""
        hyp_c = hyp.replace(" ", "")
        ref_c = ref.replace(" ", "")
        if not ref_c:
            return 0.0 if not hyp_c else 1.0
        H, R = len(hyp_c), len(ref_c)
        dp = list(range(H + 1))
        for i in range(1, R + 1):
            prev = dp[:]
            dp[0] = i
            for j in range(1, H + 1):
                if ref_c[i - 1] == hyp_c[j - 1]:
                    dp[j] = prev[j - 1]
                else:
                    dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
        return dp[H] / R