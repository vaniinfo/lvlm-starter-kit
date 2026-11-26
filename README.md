# LVLM Research Projects in Medical Imaging: How to Ramp Up

A starter kit for students in the [Duong Lab](https://sites.einsteinmed.edu/duong-lab) (Dr. Tim Q. Duong, Department of Radiology, Albert Einstein College of Medicine) who want to do research with large vision-language models (LVLMs) on medical scans. Thanks to Dr. Duong and the lab for the research onboarding it builds on.

> **Prerequisite:** first complete the [Duong Lab research onboarding VLM path](https://sites.google.com/view/research-onboarding-duong-lab/aillmvlm-path/vlm) (modules 1–6: what a VLM is, how it works, a pipeline example, and an assessment). This kit builds on top of it and applies those basics to medical imaging.

- **The guide** (models, datasets, agents, and how to approach a project): [read it online](https://vaniinfo.github.io/lvlm-starter-kit/docs/guide.html), or [`docs/guide.md`](docs/guide.md) on GitHub
- **The presentation** (interactive slides with speaker notes): [`presentation/`](presentation/README.md), online at [vaniinfo.github.io/lvlm-starter-kit/presentation](https://vaniinfo.github.io/lvlm-starter-kit/presentation/)
- **MedGemma practice, your first project** (hands-on, about one week): see [`medgemma-practice/`](medgemma-practice/README.md)
- **Start your own project:** use the [LVLM project template](https://github.com/vaniinfo/lvlm-project-template) (click **Use this template**). It has starter code and the context files an AI coding agent needs.

**Try it now:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vaniinfo/lvlm-starter-kit/blob/main/medgemma-practice/medgemma_practice.ipynb)

## What's in this repo

```
.
├── README.md              ← you are here
├── requirements.txt       ← Python packages for the first project
├── presentation/
│   ├── index.html         ← the slide deck (open in a browser; press S for speaker notes)
│   └── lib/               ← reveal.js, included so it works offline
├── docs/
│   ├── guide.md           ← the full guide
│   └── images/            ← its diagrams and charts
└── medgemma-practice/
    ├── README.md          ← step-by-step instructions
    ├── medgemma_practice.ipynb   ← MedGemma practice notebook (Colab): start here
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
