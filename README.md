# Too Many Cooks Problem Solver

## Description
One small Python tool that solves two versions of the same problem:
**many capable people, zero coordination.**

---

## Requirements
```
pip install scipy
```
(everything else is the Python standard library)

---

## How to run on VS Code 
```
python too_many_cooks.py demo             # both problems sample data
python too_many_cooks.py merge  sample_edits.json
python too_many_cooks.py assign sample_players.json
```

---

## Problem 1 — Five people editing one document
**Input:** A base document + a list of edits (`rewrite` / `delete` / `keep`),
each tagged with the editor's name. Optional per-editor trust scores.

**Decision logic (in order):**
1. **Deletion needs a majority** of everyone who touched the sentence —
   one person can't delete a paragraph another just polished.
2. **Similar rewrites are clustered** (difflib similarity ≥ 0.75) — two
   people fixing the same typo differently pool their votes into one candidate.
3. **Vote count** picks the winner among clusters (keeping the original
   is also a candidate).
4. **Tie-break #1:** average editor trust score.
5. **Tie-break #2:** the most conservative edit (closest to the original) wins.

**Output:** One merged document in one voice, plus a decision log that
explains *why* each version won.

---

## Problem 2 — Eleven football players who all want to be striker in the same team 
**Input:** Players with a skill rating per role (GK/DEF/MID/FWD) and a
preferred role (in the sample everyone prefers FWD).

**Decision logic:** The tool builds
an 11×11 skill matrix and runs the **Hungarian algorithm**
(`scipy.optimize.linear_sum_assignment`) to find the one-to-one
player→role assignment that maximizes total team skill. A tiny +0.1
preference bonus means a player gets their wish *only when the team
doesn't pay for it*.

**Output:** The optimal lineup, and a comparison in the sample data the
"everyone plays striker" team scores 714 total skill while the optimized
lineup scores 963 (**+35%**), which is the whole moral of the story.

---


## Output 
 <img width="800" height="211" alt="Output GIF File" src="https://github.com/user-attachments/assets/4d8a765a-a236-41fa-811d-5ce2205467a5" />

---
