# LVLM Research Projects in Medical Imaging: How to Ramp Up

A starter kit for students who want to do research with large vision-language models (LVLMs) on medical scans.

- **The guide** (models, datasets, agents, and how to approach a project): [`docs/guide.md`](docs/guide.md)
- **Your first project** (hands-on, about one week): see [`first-project/`](first-project/README.md)

**Try it now:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vaniinfo/lvlm-starter-kit/blob/main/first-project/first_project_colab.ipynb)

## What's in this repo

```
.
├── README.md              ← you are here
├── requirements.txt       ← Python packages for the first project
├── docs/
│   ├── guide.md           ← the full guide
│   └── images/            ← its diagrams and charts
└── first-project/
    ├── README.md          ← step-by-step instructions
    ├── first_project_colab.ipynb ← ready-to-run notebook: start here
    ├── 01_pick_images.py  ← build a small test set from ChestX-ray14
    ├── 02_ask_medgemma.py ← ask MedGemma a yes/no question about each image
    └── 03_score.py        ← compare its answers to the real labels
```

## The idea in one line

Try an existing model on a small test set you built yourself **before** training anything. That's step 1 of the "try first, train later" ladder in the guide.

## Ground rules

- **Never commit medical images or patient data to this repo.** The `data/` and `results/` folders are git-ignored for that reason.
- Use datasets only under their own terms (research use, no re-sharing).
- These models are research tools. They are not doctors, and their answers can be wrong.
