# Training the layer 2 models

Two models are fine-tuned from **CAMeLBERT-CA** (a BERT model pretrained on
Classical Arabic):

| model | what it learns | training data | file to train with | time on a T4 GPU |
|---|---|---|---|---|
| **2a, morphology** | 14 features of every unvowelled word (part of speech, clitics, person, aspect, voice, mood, gender, number, state, case, pronoun suffix) | `morph_dataset.zip`: 2.1 M Tashkeela words, labels from their vowels (GPL v2) | `train_morph.py` / `kaggle_morph.ipynb` | about 1 hour (3 epochs) |
| **2b, i'rab** | each word's i'rab role, its naḥw function, its link (المتعلَّق به، المتبوع، صاحب الحال, or a محذوف) and hidden subjects | MASAQ roles (CC BY 4.0) for the whole Qur'an, plus functions, links and hidden subjects from the Quranic Arabic Corpus treebank (~40% of the Qur'an) and Muḥyī al-Dīn ʿAbd al-Ḥamīd's iʿrāb of the Alfiyya (vols 1–2) and its shawāhid; built by `layer2b/make_dataset.py` | `train_irab.py` / `kaggle_irab.ipynb` | 15–30 minutes (10 epochs) |

> The i'rab dataset uses openly licensed or public-domain sources only. The
> QAC labels may be used for training but not redistributed: keep the Kaggle
> dataset **Private**.

Each run prints its progress, keeps the best model after every pass through
the data (epoch), and ends with the scores on held-out test data. The result
is a folder with `model.pt`, `config.json` and `tokenizer/`. Send that folder
(zipped) back, and it will be plugged into the program.

---

## Option A: on Kaggle (free GPU)

1. **Create an account** at kaggle.com and verify your phone number
   (Settings → Phone verification). Without it, GPUs and internet access are
   unavailable to notebooks.
2. **Upload the dataset.** Datasets → **New Dataset** → drop in the zip
   (`morph_dataset.zip` or `irab_dataset.zip`; Kaggle unpacks it). Name it
   `arabic-stylometrics-morph` / `arabic-stylometrics-irab`.
   For the i'rab dataset, keep the visibility **Private** (the default).
3. **Upload the notebook.** Code → **New Notebook** → File → **Import
   Notebook** → choose `kaggle_morph.ipynb` (or `kaggle_irab.ipynb`).
4. **Settings** (right-hand panel of the notebook):
   - Accelerator → **GPU T4 x2** or **GPU P100**;
   - Internet → **On** (needed once, to download CAMeLBERT-CA);
   - **Add Input** → your dataset.
5. **Run.** For a long run, use **Save Version → Save & Run All (Commit)**:
   it keeps running after you close the browser. ("Run All" in the editor
   stops when the session goes idle.)
6. **Download the result.** When the version has finished, open it →
   **Output** → download `morph_model.zip` (or `irab_model.zip`).

Kaggle gives about 30 GPU hours a week; both trainings together need about
1.5 hours.

The notebook first runs a one-minute **smoke test**. If that fails, the
error message is near the bottom of the cell's output; send it to me.

---

## Option B: on your own computer

### What you need

- **An NVIDIA graphics card with at least 8 GB of memory** (e.g. RTX 3060
  12 GB or better). With 6 GB, lower the batch size (`--batch 16` for 2a,
  `--batch 8` for 2b).
- **Apple Silicon Mac (M1–M4):** works through Apple's GPU (MPS), roughly
  3–6 times slower than a T4.
- **No GPU:** it runs on the processor too, but 2a takes days. 2b is
  feasible (a few hours).
- About 5 GB of free disk space.

### 1. Install Python and the libraries

Install **Python 3.10, 3.11 or 3.12** from python.org (on Windows, tick *Add
python.exe to PATH* during installation). Then, in a terminal (Windows:
*Command Prompt*; Mac: *Terminal*):

```bash
python -m venv stylo-env
# Windows:
stylo-env\Scripts\activate
# Mac / Linux:
source stylo-env/bin/activate
```

Install PyTorch with GPU support. Go to **pytorch.org → Get Started**, choose
your system, *Pip*, *Python* and your CUDA version, and run the command it
shows. For example, for an NVIDIA card on Windows or Linux:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

(On a Mac, or without a GPU, simply `pip install torch`.) Then:

```bash
pip install transformers
```

Check that the GPU is seen:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())"
```

`True` in the first place (NVIDIA) or the second (Mac) means the GPU will be used.

### 2. Get the code and the data

```bash
git clone https://github.com/Sentoss/Arabic-Stylometrics-Analyzer.git
cd Arabic-Stylometrics-Analyzer
```

Unzip `morph_dataset.zip` into a folder, e.g. `data/morph`, and
`irab_dataset.zip` into `data/irab`. Each folder must contain
`labels.json` and the three `.jsonl.gz` files.

### 3. Train

Check first (one minute; downloads CAMeLBERT-CA on first use):

```bash
python training/train_morph.py --data data/morph --out models/smoke --smoke
```

Then the real runs:

```bash
python training/train_morph.py --data data/morph --out models/morph --epochs 3
python training/train_irab.py  --data data/irab  --out models/irab  --epochs 10
```

Leave the terminal open until the line `done in … min` appears. The models
are in `models/morph` and `models/irab`.

### Useful options

| option | meaning | default |
|---|---|---|
| `--epochs N` | passes through the training data | 3 (2a), 10 (2b) |
| `--batch N` | sentences per step; lower it if you get "CUDA out of memory" | 32 (2a), 16 (2b) |
| `--lr X` | learning rate | 5e-5 |
| `--max-len N` | longest input in sub-word tokens | 256 |
| `--limit N` | (2a) use only the first N training segments, for a quicker trial | all |

### Troubleshooting

- **CUDA out of memory:** halve `--batch`.
- **`torch.cuda.is_available()` is False with an NVIDIA card:** the CPU-only
  PyTorch was installed; reinstall it with the command from pytorch.org.
- **Download errors for CAMeLBERT-CA:** the computer needs internet access
  the first time; afterwards the model is cached.
