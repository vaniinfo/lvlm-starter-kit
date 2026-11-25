# Your first project: test MedGemma on real chest X-rays

**Goal:** find out how well an existing medical VLM answers one simple question, using a small test set you build yourself. No training. About one week of evenings.

**The question:** "Is there fluid around the lungs (a pleural effusion) in this X-ray?" Effusion is common in the dataset and has a clear yes/no answer, which makes it a good first target.

```
1. Get the data  →  2. Build a test set  →  3. Ask the model  →  4. Score it
   (open, no         (20 images: 10 yes,     (MedGemma, one        (accuracy, missed
    sign-up)          10 no)                  question each)        cases, false alarms)
```

## Easiest way: the MedGemma practice notebook (recommended)

Everything in one place, with no downloads to your own computer:

1. Click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vaniinfo/lvlm-starter-kit/blob/main/first-project/medgemma_practice.ipynb) to open it in Colab. (Or go to [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → choose [`medgemma_practice.ipynb`](medgemma_practice.ipynb).)
   Then use **File → Save a copy in Drive** so your changes are saved.
2. Follow the "Before you start" box at the top (turn on the free GPU, get MedGemma access).
3. Run each cell with **Shift + Enter**.

The notebook downloads the X-rays itself, shows you the 20 test images, asks MedGemma about each one, scores it, and shows you the images it got wrong. An optional **Step 5** then compares four prompt styles (plain yes/no, a definition, describe-then-decide, and a 0 to 100 confidence number) on a tuning set, and checks the winner on fresh images. The scripts below do the same steps if you'd rather run things on your own machine.

## Running it with the scripts instead

### What you need

- A Google account for [Google Colab](https://colab.research.google.com) (free GPU) or a [Kaggle](https://www.kaggle.com) account.
- A free [Hugging Face](https://huggingface.co) account. MedGemma is a "gated" model: open its page (`google/medgemma-4b-it`), accept the terms, then create an access token under Settings → Access Tokens.
- About 5 GB of free disk space for one batch of images.

## Step 1: get the data (open download, no sign-up)

We use **ChestX-ray14** from the U.S. National Institutes of Health (NIH): https://nihcc.app.box.com/v/ChestXray-NIHCC

Download two things into a `data/` folder next to this file:

1. **One image batch:** `images/images_001.tar.gz` (about 5,000 X-rays). Unpack it so you get `data/images/00000001_000.png` and so on.
2. **The label file:** `Data_Entry_2017_v2020.csv` (each image's findings, such as `Effusion` or `No Finding`).

```bash
mkdir -p data
tar -xzf images_001.tar.gz -C data      # creates data/images/
mv Data_Entry_2017_v2020.csv data/
```

## Step 2: build a small test set

```bash
python 01_pick_images.py --finding Effusion --n 10
```

This picks 10 images labeled `Effusion` and 10 labeled `No Finding` from the images you downloaded, and saves the list to `data/test_set.csv`. A fixed random seed means you get the same 20 images every time, so later runs are a fair comparison.

## Step 3: ask the model

```bash
pip install -r ../requirements.txt
huggingface-cli login                    # paste your token
python 02_ask_medgemma.py --finding "a pleural effusion (fluid around the lungs)"
```

For each image, this asks MedGemma a yes/no question and saves its raw answer to `results/answers.csv`. Open that file and read some answers yourself. Seeing *how* a model is wrong teaches you a lot.

**Which GPU?** The script uses the GPU if there is one. A GPU with bfloat16 support (Colab L4 or A100, most Kaggle GPUs) works best. The free Colab T4 falls back to float16, which usually works but can occasionally give garbled answers. Without a GPU it still runs, just very slowly.

## Step 4: score it

```bash
python 03_score.py
```

You'll see a table like this, plus three numbers:

|                    | Model says yes   | Model says no    |
| ------------------ | ---------------- | ---------------- |
| **Really yes**     | caught it ✓      | **missed it** ✗  |
| **Really no**      | **false alarm** ✗ | correct ✓        |

- **Accuracy:** how many it got right out of all the images.
- **Sensitivity:** of the real cases, how many it caught. Missing a real case is often the worse mistake in medicine.
- **Specificity:** of the healthy images, how many it correctly cleared.

## What to try next

Change one thing at a time, rerun steps 3 and 4, and compare scores:

- **A different question:** pick another finding (`Cardiomegaly`, `Pneumothorax`) with `--finding` in both scripts.
- **Better instructions:** edit the question in `02_ask_medgemma.py`, or ask the model to explain before answering. That's step 2 of the ladder in the guide.
- **A different model:** try a general model, or a chest X-ray specialist like CheXagent, on the same 20 images.
- **A bigger test set:** 20 images is a demo. Real conclusions need more, so try `--n 50` or `--n 100`.

## Honest limits

- ChestX-ray14 labels were extracted automatically from reports, so some of them are wrong. Your score measures agreement with those labels, not with the truth.
- 20 images can't tell you much. One lucky or unlucky image changes the score by 5 percentage points.
- Don't commit images or results to git. The repo's `.gitignore` already blocks `data/` and `results/`.
