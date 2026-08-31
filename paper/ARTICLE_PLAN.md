# SiKDD 2026 / IS2026 — Article Plan & Reference Notes

Source ML project (code, models, results — used only as data source, not edited here):
`C:\Projects\diskFailurePrediction`

Conference site: https://aile3.ijs.si/dunja/SiKDD2026/
Multiconference site: https://is.ijs.si/
Submission portal (EasyChair): https://easychair.org/conferences/?conf=is20260 → track **"DATA MINING AND DATA WAREHOUSES – SiKDD 2026"**

---

## STEP 1 — Deadlines, Rules, Key Facts

### Critical dates
| Milestone | Date |
|---|---|
| **Paper submission deadline** | **31 August 2026** (PDF via EasyChair) |
| Notification to authors | 20 September 2026 |
| Camera-ready deadline | 25 September 2026 |
| Registration + fee payment deadline | 25 September 2026 |
| IS2026 multiconference | 5–9 October 2026, Ljubljana (hybrid) |
| **SiKDD 2026 session day** | **7 October 2026** |
| Awards ceremony | 9 October 2026, 12:00, Main Lecture Hall, IJS |

> Only ~12 days left from "today" reference in this plan until the Aug 31 deadline if written close to conference notes — **verify current date vs. deadline immediately** and prioritize accordingly.

### Submission mechanics
- Submit via **EasyChair**, select track **SiKDD 2026** (not general IS track).
- Free EasyChair account required — register early if you don't have one.
- Format: **PDF only**.
- Contact for SiKDD-specific issues: `dunja.mladenic@ijs.si`, subject line **"SiKDD paper submissions"**.
- General IS organizing contact: `is@ijs.si`.

### Paper requirements
- **Max 4 pages** (including references & appendices), English (Slovenian also allowed but we write in English).
- Options: full paper (up to 4 pages), extended abstract (up to 2 pages), or abstract (up to 1 page). >2 pages = treated as full paper.
- Abstract **mandatory**.
- CCS concepts + keywords: **required if >2 pages** (our paper will need them).
- Must use the **official template** (Word or LaTeX) — found locally in `2026/` subfolder:
  - `IS-Word_template_2026.docx`
  - `sample-is-2026.tex` + `acmart.cls` + `sample-base.bib` (LaTeX/BibLaTeX route)
- Template is **ACM `acmart` sigconf class**, modified: `\geometry{a4paper}`, `printccs=false, printacmref=false`, uses **biblatex+biber** (not natbib/bibtex).
- **No modification of margins, fonts, spacing** — papers with template modifications get returned for revision.
- No page numbers in submitted paper.
- Title: proper Title Case capitalization (English rule — common author mistake per instructions).
- Every figure needs a `\Description{}` (accessibility alt-text, <2000 chars, must not just repeat the caption).
- Table captions **above** table; figure captions **below** figure.
- `\acmDOI{}` left empty at submission — filled in only at camera-ready stage once chairs issue a DOI.
- Optional **AI self-review** before submitting: prompt available at `prompt_is_conference.txt` (already fetched — can run our draft through an LLM with this reviewer persona before submission to catch structural/completeness/language issues).

### Registration & fees
- One author must register **per accepted paper** (not per author).
- Fee: **200 EUR full / 100 EUR student**.
- Registration form: https://is.ijs.si/?page_id=12560
- Presentation: 10–20 min oral; hybrid (in-person, Zoom/Teams, or pre-recorded by arrangement).

### Conference themes (pick the best-fitting one(s) for framing)
Statistical Data Analysis & Causal Inference · Data/Text/Graph/Web/Multimedia Mining · **LLMs & Generative AI** · Link Detection/Social Network Analysis/Knowledge Graphs · NLP & Semantic Tech · **Sensor Data Analytics, Stream Mining, and Edge AI** · **Explainable AI (XAI), Ethics, Trustworthy ML** · Modeling Complex Systems & AI for Science · **Applications**

→ Our disk-failure-prediction project fits best under **Sensor Data Analytics/Edge AI**, **Applications**, and secondarily **XAI/Trustworthy ML** (multi-model fusion + interpretable AHI score).

---

## STEP 2 — Writing Plan (angle confirmed)

### 2.1 Angle (decided)
**Hybrid framing** — technical enough to briefly cover all 4 models + dataset + pipeline (so it reads as a real scientific/technical contribution, not just a position paper), but wrapped in an applications/big-picture narrative:
- What problem this solves (reactive vs. proactive drive replacement, data loss/downtime cost, blind spots of manufacturer SMART thresholds — see Related Work).
- Where it's useful (data centers, NAS/edge devices, backup infra — offline, lightweight inference).
- Pros/cons of the ensemble+fusion approach vs. a single model (robustness/interpretability vs. added complexity).
- Technical core: dataset, 4 models briefly, AHI fusion, results table.

This maps to conference tracks **Applications** + **Sensor Data Analytics/Edge AI**, with a secondary nod to **Explainable/Trustworthy ML** (AHI as an auditable, componentized score rather than a black-box single number).

### 2.2 Related work landscape (found via search, to cite/paraphrase — verify each before citing)
- Murray et al., *"Machine Learning Methods for Predicting Failures in Hard Drives: A Multiple-Instance Application"*, JMLR 2005 — foundational SMART+ML paper, established manufacturer threshold FDR (3–10%) baseline.
- Lu et al., *"Making Disk Failure Predictions SMARTer!"*, USENIX FAST 2020 — large-scale (380k disks) SMART+performance+location fusion study, 0.95 F-measure.
- Xu et al., *"Layerwise Perturbation-Based Adversarial Training for Hard Drive Health Degree Prediction"*, arXiv:1809.04188 — health-degree (not just binary) prediction, semi-supervised, imbalance-aware.
- *"Cost aware LSTM model for predicting hard disk drive failures based on extremely imbalanced S.M.A.R.T. sensors data"*, ScienceDirect 2023 — explicitly addresses Backblaze class imbalance (11,501:1), directly relevant to our balancing approach.
- *"Proactive Drive Failure Prediction for Cloud Storage System Through Semi-Supervised Learning"*, IEEE TDSC 2023 — semi-supervised + interpretability angle, good contrast point for our supervised+unsupervised fusion (AHI).

→ Use these to frame the "niche": most prior work picks **one** paradigm (supervised classifier OR anomaly detection OR clustering); our contribution is **fusing all three paradigms into one interpretable componentized score (AHI)** rather than a single black-box output.

### 2.3 Section skeleton (~4 pages total, IMRaD-based, adapted per Related Work findings above)
1. **Title + Abstract** (~150–200 words) — problem, approach (4-model ensemble + AHI fusion), headline result (89.1% recall / 0.929 ROC-AUC), where it's deployable.
2. **Introduction** (~0.5 page) — cost of drive failure (data loss/downtime), manufacturer SMART thresholds' low FDR (3–10%, cite Murray05), gap: most ML work picks one paradigm; our contribution bullets (3–4 lines, one-line each).
3. **Related Work** (~0.4 page) — the 5 refs above, positioned to expose the "fuse multiple paradigms into one interpretable score" niche.
4. **Data & Methodology** (~1.3 pages) — Backblaze 2025 dataset (32M+ rows, 4,414 failures, 365 daily files), preprocessing/balancing, then **briefly** each of the 4 models (RF baseline; AE anomaly; bottleneck AE+classifier; UMAP+HDBSCAN), then AHI weighted-RMS fusion formula + weight rationale.
5. **Results** (~0.9 page) — performance table (ROC-AUC/Recall/F1/AHI weight per model, from README), 1 key figure (likely AHI formula figure or bottleneck architecture — pick the one that best supports the "fusion adds value" claim), 2–3 sentence result interpretation.
6. **Discussion — Use Cases, Pros/Cons, Limitations** (~0.5 page) — deployment scenarios (data center fleet monitoring, NAS/edge, offline inference), pros (interpretable componentized score, no cloud dependency, robustness via ensemble) vs. cons (current-state not time-to-failure, added system complexity vs. single model, no location/environmental features per Lu20), honest limitation: single-vendor/point-in-time snapshot dataset.
7. **Conclusion & Future Work** (~0.2 page) — summarize, future: time-to-failure horizon, location/environmental features, journal extension.
8. **References** (~8–12, BibLaTeX in `sample-base.bib` style).

### 2.4 Assets to pull from `diskFailurePrediction` (read-only source)
- Performance tables from `@/C:/Projects/diskFailurePrediction/README.md:44-49`
- Candidate figures from `Graphs/`: `hir_formula.png`, `nn_classification.png`, `umap_hdbscan.png`, `classification.png` — must pick only 1–2 given space budget
- AHI formula weights and fusion rationale from `srcML/hir_final.py` and README AHI section

### 2.5 Process
1. Lock the angle (Step 2.1).
2. Draft in `sample-is-2026.tex` (copy → rename), section by section.
3. Fill CCS concepts + keywords.
4. Insert real tables/figures + rewrite captions/descriptions.
5. Self-check with the AI reviewer prompt (`prompt_is_conference.txt`) before submission.
6. Proofread language, verify Title Case, verify template untouched (margins/fonts).
7. Export PDF, submit via EasyChair (SiKDD 2026 track) before **31 Aug 2026**.
8. After notification (20 Sep): revise, insert DOI placeholder instructions, resubmit camera-ready by **25 Sep 2026**, register + pay fee by same date.
