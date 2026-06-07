# Tarteel PoC — Quran Recitation Recognition Pipeline

A Python proof-of-concept reproducing the Tarteel AI pipeline:
**audio → Whisper transcription → Quran text alignment with diacritics**.

---

## Architecture

```
tarteel-poc/
├── cli.py                  ← entry point (file mode + realtime mode)
├── requirements.txt
│
├── core/
│   ├── model.py            ← Whisper model loader + transcription
│   ├── audio.py            ← audio loading, resampling, chunking
│   └── alignment.py        ← dynamic-programming word alignment engine
│
├── quran/
│   ├── loader.py           ← downloads + caches Quran JSON (with tashkeel)
│   └── data/
│       └── quran.json      ← auto-downloaded on first run
│
├── pipeline/
│   └── runner.py           ← orchestrates all three stages, rich CLI output
│
└── realtime/
    └── stream.py           ← VAD-based mic streaming (Phase 3)
```

---

## Pipeline Flow

```
Audio File / Mic
      │
      ▼
 AudioLoader          — librosa: resample to 16kHz mono, normalise
      │
      ▼
 TarteelModel         — tarteel-ai/whisper-base-ar-quran (HuggingFace)
      │                 outputs: raw Arabic text WITHOUT diacritics
      ▼
 QuranAligner         — normalise both texts (strip diacritics, alef variants)
      │                 fuzzy-search the 6,236-verse index (rapidfuzz)
      │                 DP word-level alignment (classic WER algorithm)
      │                 map each word back to its diacritic form
      ▼
 AlignmentResult      — WER, CER, word-level correct/substitution/deletion/insertion
                        ground truth with tashkeel highlighted per word
```

---

## Setup

### 1. Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# For real-time mic mode — also install system lib:
# macOS:  brew install portaudio
# Linux:  sudo apt-get install portaudio19-dev python3-dev
```

### 2. First run (auto-downloads model + Quran data)

On first run the pipeline will:
- Download `tarteel-ai/whisper-base-ar-quran` (~290 MB) from HuggingFace.
- Download the full Quran with tashkeel from `api.alquran.cloud` (~2 MB) and cache it at `quran/data/quran.json`.

Both are cached locally; subsequent runs are fast and fully offline.

---

## Usage

### File mode

```bash
# Auto-detect which verse is being recited
python cli.py file path/to/recitation.wav

# Align against a specific verse (Surah 1, Ayah 1 = Al-Fatiha opening)
python cli.py file recitation.wav --surah 1 --ayah 1

# Search within a surah (e.g. Al-Baqarah)
python cli.py file recitation.wav --surah 2

# With word-level timestamps from Whisper
python cli.py file recitation.wav --surah 1 --ayah 1 --timestamps

# Use GPU
python cli.py file recitation.wav --device cuda
```

### Real-time mic mode

```bash
# Auto-detect verses while you recite
python cli.py realtime

# Lock to Al-Fatiha
python cli.py realtime --surah 1 --ayah 1

# Higher VAD aggressiveness (filters background noise better)
python cli.py realtime --vad 3
```

### Python API

```python
from pipeline import TarteelPipeline

pipe = TarteelPipeline()

# Process a file
result = pipe.run("recitation.wav", surah=1, ayah=1)

# Print formatted result
pipe.print_result(result)

# Access data programmatically
print(result.reference_with_diacritics)   # full ground truth with tashkeel
print(result.wer)                          # word error rate
for wa in result.word_alignments:
    print(wa.ref_word_with_diac, wa.status)   # per-word status
```

---

## Output Example

```
══════════════════ Alignment Result ══════════════════

┌─ Verse Match ────────────────────────────────────────┐
│ Surah 1, Ayah 2                                       │
│                                                       │
│ Ground truth (with tashkeel):                         │
│   الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ             │
│                                                       │
│ Your recitation (as transcribed):                     │
│   الحمد لله رب العالمين                               │
└───────────────────────────────────────────────────────┘

╭──────────────────╮
│ Match Score      │ 98 / 100    │
│ Word Accuracy    │ 100.0%      │
│ WER              │ 0.0%        │
│ CER              │ 0.0%        │
│ Correct Words    │ 4 / 4       │
╰──────────────────╯

Word-level alignment:
┌──────────────────────────────────────────────────────┐
│ الْحَمْدُ  لِلَّهِ  رَبِّ  الْعَالَمِينَ           │
│ (green = correct, yellow = substitution,              │
│  red = deletion, magenta = insertion)                 │
└───────────────────────────────────────────────────────┘
```

---

## AlignmentResult fields

| Field | Type | Description |
|-------|------|-------------|
| `surah` | int | Matched surah number |
| `ayah` | int | Matched ayah number |
| `hypothesis` | str | Whisper output (no diacritics) |
| `reference_plain` | str | Ground truth stripped of diacritics |
| `reference_with_diacritics` | str | Full ground truth with tashkeel |
| `word_alignments` | list[WordAlignment] | Per-word alignment details |
| `wer` | float | Word Error Rate (0.0–1.0) |
| `cer` | float | Character Error Rate (0.0–1.0) |
| `match_score` | float | Fuzzy match confidence (0–100) |
| `correctly_recited_words` | list[str] | Words recited correctly (with diacritics) |
| `error_words` | list[str] | Words with errors |

### WordAlignment fields

| Field | Description |
|-------|-------------|
| `hyp_word` | What Whisper heard |
| `ref_word` | Ground truth (no diacritics) |
| `ref_word_with_diac` | Ground truth (with diacritics) |
| `status` | `correct` / `substitution` / `insertion` / `deletion` |
| `similarity` | Character-level similarity score 0–1 |

---

## Phase Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Done | Model loading, audio preprocessing, Quran data download |
| 2 | ✅ Done | Alignment engine, full pipeline, rich CLI output |
| 3 | ✅ Done | Real-time mic streaming with VAD |

---

## Notes for Graduation Project

- The model outputs **undiacritised** Arabic. The alignment engine strips diacritics from the ground truth for comparison, then maps each matched word back to its original diacritised form — this is the core of the "diacritics assignment" problem.
- To improve on Tarteel, you could fine-tune on a larger/more diverse Quranic dataset, or experiment with `whisper-small-ar-quran` for better accuracy.
- The alignment module (`core/alignment.py`) is where most academic contribution lives — you could replace the DP algorithm with a beam-search lattice aligner or integrate an Arabic-aware edit distance.