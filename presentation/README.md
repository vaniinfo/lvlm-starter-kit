# Presentation

An interactive slide deck version of the [guide](../docs/guide.md), built with [reveal.js](https://revealjs.com) (included in `lib/`, so it works without internet).

**Open it online:** https://vaniinfo.github.io/lvlm-starter-kit/presentation/
(This link works once GitHub Pages is turned on for the repo. See "Hosting" below.)

## Presenting

| Key | What it does |
| --- | --- |
| → / ← | next / previous slide (some slides have more below them: use ↓) |
| **S** | opens the **speaker view**: current slide, next slide, timer and your notes |
| **F** | full screen |
| **Esc** | overview of all slides |

- With a projector or second screen: put the slides on the big screen and keep the speaker view on your laptop.
- If nothing happens when you press **S**, allow pop-ups for the page in your browser.
- Speaker notes are visible to anyone who views the page's code, so don't put anything private in them.

## Interactive slides

- **Quick checks:** three multiple-choice questions. Ask the room first, then click an answer to show whether it's right and why.
- **Move the cutoff:** drag the slider to see sensitivity go up as specificity goes down (made-up example scores).
- **Live demo:** a button that opens the MedGemma practice notebook in Colab.

## Printing with notes

Open the deck with `?print-pdf&showNotes=true` at the end of the address, for example
`https://vaniinfo.github.io/lvlm-starter-kit/presentation/?print-pdf&showNotes=true`,
then use the browser's **Print → Save as PDF**. Each slide prints with its notes underneath, which is handy for rehearsing.

## Hosting (one-time)

On GitHub, open the repo → **Settings → Pages** → under "Build and deployment" choose **Deploy from a branch**, branch **main**, folder **/ (root)** → **Save**. After a minute or two the link above goes live.

## Editing

Everything is in `index.html`: each slide is a `<section>`, and its speaker notes are in the `<aside class="notes">` inside it. The diagrams come from `../docs/images/`, the same files the guide uses.
