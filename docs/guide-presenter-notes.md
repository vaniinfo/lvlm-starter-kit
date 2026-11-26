# Presenter notes: LVLM guide walkthrough

For presenting **https://vaniinfo.github.io/lvlm-starter-kit/docs/guide.html** while these notes are open on a
second screen. Each part says what to have **on screen**, what to **say**, and what to **ask**.

**Before you start:** open the guide in the browser, zoom to about 125% (Cmd/Ctrl and +), and close other tabs.
Use the Agenda links to jump between sections instead of long scrolling. Have the
[MedGemma practice notebook in Colab](https://colab.research.google.com/github/vaniinfo/lvlm-starter-kit/blob/main/medgemma-practice/medgemma_practice.ipynb)
open in a second tab for the demo.

## Timing (about 40 minutes + questions)

| Part | Section | Minutes |
|---|---|---|
| 1 | Title and Before you start | 3 |
| 2 | 1. Three kinds of models, VLM vs. LVLM | 4 |
| 3 | 2. X-ray vs. CT vs. MRI | 4 |
| 4 | 3. Try first, train later, and my path | 6 |
| 5 | Your first project + demo | 4 |
| 6 | Better prompts + two real traps | 5 |
| 7 | 4. How results are scored | 4 |
| 8 | 5. Available models + 6. Agents | 4 |
| 9 | 7. My two projects | 3 |
| 10 | 8. Datasets, 9. Ethics, 10. Research question | 3 |
| — | 11. Advanced training: **skip**, mention only | 0–1 |
| 11 | Wrap-up: next steps and discussion | 2+ |

Short on time? Cut parts 8 and 10 to one sentence each; the guide is there for them to read later.

---

## 1. Title and "Before you start" (3 min)

**On screen:** the title, then the three-box diagram.

**Say:**
- "This guide is for students in Dr. Duong's lab who want to do research with AI that reads medical scans."
- Point to the three boxes, left to right:
  - **Box 1:** "First, the lab's onboarding VLM path. That's the basics: what a VLM is and how it works."
  - **Box 2:** "Then this guide. It follows my own path, X-ray, then CT, now MRI, and shares what I learned so you can move faster."
  - **Box 3:** "Then your own project, starting from the project template."
- "Boxes 1 and 3 are clickable. You don't need to open them now."

**Ask:** "Who has finished the onboarding VLM path?" (If few hands: "That's fine, it's the first thing to do after today.")

---

## 2. Three kinds of medical AI models (4 min)

**On screen:** the three-kinds diagram, then the "Same X-ray, two kinds of model" diagram.

**Say:**
- "Same scan, three kinds of answer. A **classifier** says *what*: pneumonia, yes or no. A **segmentation** model says *where*: it outlines the lungs. A **vision-language model** answers in *words*."
- "The older two are often more accurate at their one job, but they can't explain anything. VLMs talk, but they're less precise. Good projects combine all three."
- On the VLM vs. LVLM diagram: "A plain VLM can only score captions you hand it. It never writes new words. A *large* VLM, an LVLM, like MedGemma, answers your question in its own words. More useful, but it can sound sure even when it's wrong."
- "Why medicine cares: billions of scans a year, not enough radiologists."

**Ask:** "If you had to trust one of these three for a life-or-death call, which one, and why?"

---

## 3. Three categories of scans (4 min)

**On screen:** the X-ray / CT / MRI diagram and table.

**Say:**
- "**X-ray** is one photo. **CT** is a loaf of bread sliced into hundreds of slices. **MRI** is slices taken several ways, sometimes a video of the beating heart."
- "Each step adds a dimension, so each is harder for AI. And there's less training data at each step."
- "**The 3D CT gap:** most popular models, including MedGemma, were built for flat pictures. They flip through a sample of a CT's slices instead of understanding the whole 3D volume. That gap is why CT is a focus of our work."

**Ask:** "Why do you think there's so much more X-ray data than MRI data?" (Answer if needed: X-rays are the most common scan, and MRI is expensive and comes in many settings.)

---

## 4. Try first, train later + my path (6 min)

**On screen:** "Pick a starter topic", then the "my own path" diagram, then the ladder.

**Say:**
- "**Start with chest X-ray**: one image, runs on a free GPU, lots of models and open data. Two honest cautions: start with fully open data like NIH ChestX-ray14, and know that X-ray is crowded, so it's for learning, not for standing out."
- On the path diagram: "This is what I did. X-ray first: that taught me the basics. Then lung CT, which is my current project. And I've just started on cardiac MRI."
- On cardiac MRI, keep it short: "It's a pipeline around a published model called CineMA. It outlines the heart's chambers on every frame and measures how much blood the heart pumps per beat. The honest part: no accuracy measured yet, because there are no expert outlines, and models agreeing with each other doesn't prove they're right."
- On the ladder: "**Step 0** is the most important: a small test set of your own images with the right answers. Then climb only if the step below isn't good enough: try as-is, better prompts, add tools, then training."
- Analogy: "Taste the store-bought sauce first, then season it, and only cook from scratch if nothing else works."

**Ask:** "Why build the test set *before* trying any model?" (Answer: otherwise you can't tell whether a change helped.)

---

## 5. Your first project + demo (4 min)

**On screen:** the first-project diagram. Then switch to the Colab tab.

**Say:**
- "This is step 1 of the ladder, done for real. One question, 'Is there fluid around the lungs?', on 20 chest X-rays."
- In Colab, scroll through quickly, don't run it live: "Download the data, pick 10 sick and 10 healthy, ask MedGemma, score it. About a week of work, on a free GPU."
- "Twenty images is a demo, not a result: one image moves the score by 5 points."

**If asked "do I need to code?":** "A little Python helps, but the notebook is ready to run. You'll learn by changing one thing at a time."

---

## 6. Better prompts + two real traps (5 min)

**On screen:** the four-prompt-styles diagram, the blue banner, then the "Two real prompt traps" diagram.

**Say:**
- "The wording alone can change the score. That's step 2 of the ladder."
- Read the banner: "**Prompt testing is a critical step: expect many rounds to reach the accuracy you need.**"
- **Trap 1**, point to the red **right** and **Biased**: "R and L mark the patient's right and left. The fluid is on the left. When the prompt said 'locate **right** pleural effusion', the model boxed the right side anyway. It went with what we told it, not the image. When we hid the side, it boxed the correct side. So never put the answer you're checking into the prompt."
- **Trap 2**, point to the red **a** / **A**: "On a lung CT, lowercase 'atelectasis' got Yes. Capital 'Atelectasis' got No. Same scan, same everything else, with thinking on or off. Models read text as tokens, so one capital letter can look like a different word."
- "Each is one case: a warning sign to test for, not an error rate."

**Ask:** "What other small wording change do you think could flip an answer?" (Good answers: synonyms, negation like "no effusion", order of options, adding "likely".)

---

## 7. How results are scored (4 min)

**On screen:** the four-boxes scoring diagram.

**Say:**
- "Every yes/no answer lands in one of four boxes: **caught it**, **missed it**, **false alarm**, or **correctly cleared**."
- "**Accuracy can fool you.** If 5 in 100 X-rays show disease, a model that always says 'healthy' scores 95% while catching nothing."
- "**Sensitivity**: of the sick, how many caught. **Specificity**: of the healthy, how many cleared. Making the model say yes more easily trades one for the other."
- "**AUC** compares models fairly across all cutoffs. I used it to compare MedGemma, CheXagent and MAIRA-2."

**Ask:** "For a screening tool, which mistake is worse: missing a sick patient, or a false alarm?" (Usually missing a sick patient, but false alarms cost time and worry.)

---

## 8. Available models + agents (4 min)

**On screen:** the models map, then the agent diagram.

**Say:**
- "Three layers: **all-rounders** like MedGemma, like a family doctor; **specialists** for one scan type, like CheXagent for chest X-ray; and **agents** on top."
- "A surprise: in one 2025 test, general GPT-4o beat both CheXagent and MedGemma on a question-answering task. A specialist isn't automatically the winner."
- "MRI has the fewest models: a wide-open area."
- On agents: "A single VLM is one doctor glancing at a scan. An agent is the attending physician running a team: it calls specialist tools, compares their answers, then decides. And every step is logged, so a doctor can check how it got there."

---

## 9. My two projects (3 min)

**On screen:** the chest X-ray project diagram, then the lung CT diagram.

**Say:**
- **Chest X-ray:** "One VLM writes the draft, a second checks it, like a resident's report checked by a senior doctor. A radiologist grades the result."
- **Lung CT:** "Each tool does what it's best at: segmentation measures *where* and *how much*, classifiers say *what*, a VLM *describes*. Everything goes into one evidence table, and the report may only claim what the table supports. It's allowed to say 'cannot tell'."
- "One finding shaped the design: general VLMs answered 'absent' to every finding on our CT scans. So we show them but don't trust them, and we're trying NV-Reason-CT, a model built to read the whole 3D scan."

---

## 10. Datasets, ethics, research question (3 min)

**On screen:** dataset sizes chart, the access chart, then the ethics diagram, then the spot-the-gap checklist.

**Say:**
- **Datasets:** "X-ray has hundreds of thousands of examples, CT tens of thousands, MRI far fewer, in the same order as how mature the models are."
- "**Public doesn't mean one-click.** MIMIC-CXR needs a CITI ethics course and approval. Start with fully open sets."
- **Ethics:** "Real patients' scans. The rules apply from your very first project. And if a result looks too good to be true, check that the same patient isn't in both your training and test images."
- **Research question:** "Small, specific, and testable with data you can actually get. Run your first project, then use the checklist to spot a gap. Repeating someone's result on new data is real research too."

---

## 11. Advanced training (skip)

**Say only:** "Section 11 covers training: LoRA, fine-tuning, reinforcement learning. That's steps 4 and 5 of the ladder. Come back to it after you've tried the first three steps on your own test set."

---

## 12. Wrap-up (2 min + discussion)

**On screen:** scroll back up to "Before you start" (the three boxes).

**Say:**
- "Three things to do this week:
  1. Finish the onboarding VLM path.
  2. Run the MedGemma practice notebook.
  3. When you're ready, click **Use this template** on the project template, and run its exploration notebook in Colab."
- "Everything is linked from the guide: this page, the slides and the repo."

**Discussion** (section 13 of the guide; pick one or two):
1. If an AI and a doctor disagree about a scan, who should decide, and why?
2. Why might a general AI sometimes beat a medical-specialist AI?
3. What could go wrong if an AI is trained mostly on scans from one country or one hospital?

---

## If someone asks…

| Question | Short answer |
|---|---|
| "Can this AI diagnose patients?" | No. These are research experiments, not clinical tools. Every model here is for research use only. |
| "Do I need a GPU?" | No. Free Colab GPUs are enough for the first project and the exploration notebook. |
| "Which model should I start with?" | MedGemma on chest X-ray: it's what most new research compares against. |
| "How long until I have a real result?" | The first project takes about a week. A real result needs 25 + 25 images to tune on and a separate set to confirm on. |
| "Is the cardiac MRI work done?" | No, it just started. The left ventricle looks consistent; the right ventricle isn't usable yet, and no accuracy has been measured. |
| "Why did the capital A change the answer?" | Models split text into pieces called tokens, and "Atelectasis" and "atelectasis" can become different pieces. We tested it with thinking on and off, but it's still one case. |
