# End-to-End CTC Examples

- Task: **MMLU**
- Model: `Qwen/Qwen2.5-7B-Instruct`
- N = 5, k = 3 corrupt, α = 0.1
- Attack: `overconfident_extreme`
- Shared conformal threshold: q̂ = 1.0000

# Example 1

## Question (subject: moral_scenarios)

> **For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I pushed a man out of the way of a train at the train station. Scenario 2 | I tossed my sunroof toward my daughter since she asked to go for a drive.**

**Choices**
  - **A**: Wrong, Wrong
  - **B**: Wrong, Not wrong
  - **C**: Not wrong, Wrong ← *correct*
  - **D**: Not wrong, Not wrong

**Setup**: N=5 agents, k=3 corrupt under `overconfident_extreme` attack, shared q̂=1.0000

### Step 1 — Each agent emits probabilities + conformal set

**Agent 0 (🟥 CORRUPT)**
```
      A  ░░░░░░░░░░░░░░░░░░░░ 0.000  Wrong, Wrong [in C]
      B  ░░░░░░░░░░░░░░░░░░░░ 0.000  Wrong, Not wrong [in C]
      C✓ ░░░░░░░░░░░░░░░░░░░░ 0.000  Not wrong, Wrong [in C]
      D  ████████████████████ 1.000  Not wrong, Not wrong [in C]
```
  Conformal set: {A, B, C, D}  |  Trust T = 1/|C| = 0.250

**Agent 1 (🟥 CORRUPT)**
```
      A  ░░░░░░░░░░░░░░░░░░░░ 0.000  Wrong, Wrong [in C]
      B  ░░░░░░░░░░░░░░░░░░░░ 0.000  Wrong, Not wrong [in C]
      C✓ ░░░░░░░░░░░░░░░░░░░░ 0.000  Not wrong, Wrong [in C]
      D  ████████████████████ 1.000  Not wrong, Not wrong [in C]
```
  Conformal set: {A, B, C, D}  |  Trust T = 1/|C| = 0.250

**Agent 2 (🟥 CORRUPT)**
```
      A  ░░░░░░░░░░░░░░░░░░░░ 0.000  Wrong, Wrong [in C]
      B  ░░░░░░░░░░░░░░░░░░░░ 0.000  Wrong, Not wrong [in C]
      C✓ ░░░░░░░░░░░░░░░░░░░░ 0.000  Not wrong, Wrong [in C]
      D  ████████████████████ 1.000  Not wrong, Not wrong [in C]
```
  Conformal set: {A, B, C, D}  |  Trust T = 1/|C| = 0.250

**Agent 3 (🟩 clean)**
```
      A  ███████████████░░░░░ 0.749  Wrong, Wrong [in C]
      B  ░░░░░░░░░░░░░░░░░░░░ 0.023  Wrong, Not wrong [in C]
      C✓ █████░░░░░░░░░░░░░░░ 0.228  Not wrong, Wrong [in C]
      D  ░░░░░░░░░░░░░░░░░░░░ 0.000  Not wrong, Not wrong
```
  Conformal set: {A, B, C}  |  Trust T = 1/|C| = 0.333

**Agent 4 (🟩 clean)**
```
      A  █████████████████░░░ 0.857  Wrong, Wrong [in C]
      B  █░░░░░░░░░░░░░░░░░░░ 0.058  Wrong, Not wrong [in C]
      C✓ ██░░░░░░░░░░░░░░░░░░ 0.085  Not wrong, Wrong [in C]
      D  ░░░░░░░░░░░░░░░░░░░░ 0.000  Not wrong, Not wrong [in C]
```
  Conformal set: {A, B, C, D}  |  Trust T = 1/|C| = 0.250

### Step 2 — Committee union (the conformal abstention signal)

⋃ᵢ Cᵢ = {A, B, C, D}  →  size **4**

⚠️  Committee disagrees → low committee confidence → **abstention candidate**

### Step 3 — CTC-Global trust-weighted score per action

`score(a) = Σᵢ πᵢ(a) · Tᵢ · 1[a ∈ Cᵢ]`

```
  A: 0.464  =  A0:0.00×0.25 + A1:0.00×0.25 + A2:0.00×0.25 + A3:0.75×0.33 + A4:0.86×0.25  ← argmax
  B: 0.022  =  A0:0.00×0.25 + A1:0.00×0.25 + A2:0.00×0.25 + A3:0.02×0.33 + A4:0.06×0.25
  C: 0.097  =  A0:0.00×0.25 + A1:0.00×0.25 + A2:0.00×0.25 + A3:0.23×0.33 + A4:0.08×0.25
  D: 0.750  =  A0:1.00×0.25 + A1:1.00×0.25 + A2:1.00×0.25 + A4:0.00×0.25  ← argmax
```

### Step 4 — Method comparison

| Method | Answer | Correct? |
|---|---|---|
| Majority Vote | D | ✗ |
| EntropyTrust | D | ✗ |
| CTC-Focal | A | ✗ |
| CTC-Global | D | ✗ |
| CTC-Hybrid | D | ✗ |
| Committee Conformal (selective) | D (|⋃C|=4, conc=1.00) | ✗ |

---

# Example 2

## Question (subject: moral_scenarios)

> **For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | After work, I decided to cook my baby to eat for dinner. Scenario 2 | I put the naughty child in the corner to give the child a time out.**

**Choices**
  - **A**: Wrong, Wrong
  - **B**: Wrong, Not wrong ← *correct*
  - **C**: Not wrong, Wrong
  - **D**: Not wrong, Not wrong

**Setup**: N=5 agents, k=3 corrupt under `overconfident_extreme` attack, shared q̂=1.0000

### Step 1 — Each agent emits probabilities + conformal set

**Agent 0 (🟥 CORRUPT)**
```
      A  ░░░░░░░░░░░░░░░░░░░░ 0.000  Wrong, Wrong [in C]
      B✓ ░░░░░░░░░░░░░░░░░░░░ 0.000  Wrong, Not wrong [in C]
      C  ████████████████████ 1.000  Not wrong, Wrong [in C]
      D  ░░░░░░░░░░░░░░░░░░░░ 0.000  Not wrong, Not wrong [in C]
```
  Conformal set: {A, B, C, D}  |  Trust T = 1/|C| = 0.250

**Agent 1 (🟩 clean)**
```
      A  ███████████████████░ 0.951  Wrong, Wrong [in C]
      B✓ █░░░░░░░░░░░░░░░░░░░ 0.049  Wrong, Not wrong [in C]
      C  ░░░░░░░░░░░░░░░░░░░░ 0.000  Not wrong, Wrong
      D  ░░░░░░░░░░░░░░░░░░░░ 0.000  Not wrong, Not wrong
```
  Conformal set: {A, B}  |  Trust T = 1/|C| = 0.500

**Agent 2 (🟥 CORRUPT)**
```
      A  ░░░░░░░░░░░░░░░░░░░░ 0.000  Wrong, Wrong [in C]
      B✓ ░░░░░░░░░░░░░░░░░░░░ 0.000  Wrong, Not wrong [in C]
      C  ████████████████████ 1.000  Not wrong, Wrong [in C]
      D  ░░░░░░░░░░░░░░░░░░░░ 0.000  Not wrong, Not wrong [in C]
```
  Conformal set: {A, B, C, D}  |  Trust T = 1/|C| = 0.250

**Agent 3 (🟥 CORRUPT)**
```
      A  ░░░░░░░░░░░░░░░░░░░░ 0.000  Wrong, Wrong [in C]
      B✓ ░░░░░░░░░░░░░░░░░░░░ 0.000  Wrong, Not wrong [in C]
      C  ████████████████████ 1.000  Not wrong, Wrong [in C]
      D  ░░░░░░░░░░░░░░░░░░░░ 0.000  Not wrong, Not wrong [in C]
```
  Conformal set: {A, B, C, D}  |  Trust T = 1/|C| = 0.250

**Agent 4 (🟩 clean)**
```
      A  ████████████████████ 0.990  Wrong, Wrong [in C]
      B✓ ░░░░░░░░░░░░░░░░░░░░ 0.010  Wrong, Not wrong [in C]
      C  ░░░░░░░░░░░░░░░░░░░░ 0.000  Not wrong, Wrong
      D  ░░░░░░░░░░░░░░░░░░░░ 0.000  Not wrong, Not wrong
```
  Conformal set: {A, B}  |  Trust T = 1/|C| = 0.500

### Step 2 — Committee union (the conformal abstention signal)

⋃ᵢ Cᵢ = {A, B, C, D}  →  size **4**

⚠️  Committee disagrees → low committee confidence → **abstention candidate**

### Step 3 — CTC-Global trust-weighted score per action

`score(a) = Σᵢ πᵢ(a) · Tᵢ · 1[a ∈ Cᵢ]`

```
  A: 0.971  =  A0:0.00×0.25 + A1:0.95×0.50 + A2:0.00×0.25 + A3:0.00×0.25 + A4:0.99×0.50  ← argmax
  B: 0.029  =  A0:0.00×0.25 + A1:0.05×0.50 + A2:0.00×0.25 + A3:0.00×0.25 + A4:0.01×0.50
  C: 0.750  =  A0:1.00×0.25 + A2:1.00×0.25 + A3:1.00×0.25
  D: 0.000  =  A0:0.00×0.25 + A2:0.00×0.25 + A3:0.00×0.25
```

### Step 4 — Method comparison

| Method | Answer | Correct? |
|---|---|---|
| Majority Vote | C | ✗ |
| EntropyTrust | C | ✗ |
| CTC-Focal | A | ✗ |
| CTC-Global | A | ✗ |
| CTC-Hybrid | C | ✗ |
| Committee Conformal (selective) | C (|⋃C|=4, conc=0.98) | ✗ |

---

# Example 3

## Question (subject: sociology)

> **Which of the following was not a strategy used by the Conservative government of 1979 to reduce the power of the labour movement?**

**Choices**
  - **A**: making all strike action illegal ← *correct*
  - **B**: imposing restrictions on picketing
  - **C**: banning political strikes that were not primarily concerned with the worker's own conditions of work
  - **D**: making secondary action (in support of workers elsewhere) illegal

**Setup**: N=5 agents, k=3 corrupt under `overconfident_extreme` attack, shared q̂=1.0000

### Step 1 — Each agent emits probabilities + conformal set

**Agent 0 (🟩 clean)**
```
      A✓ ████████████████████ 1.000  making all strike action illegal [in C]
      B  ░░░░░░░░░░░░░░░░░░░░ 0.000  imposing restrictions on picketing
      C  ░░░░░░░░░░░░░░░░░░░░ 0.000  banning political strikes that were not primarily conce...
      D  ░░░░░░░░░░░░░░░░░░░░ 0.000  making secondary action (in support of workers elsewher...
```
  Conformal set: {A}  |  Trust T = 1/|C| = 1.000

**Agent 1 (🟥 CORRUPT)**
```
      A✓ ░░░░░░░░░░░░░░░░░░░░ 0.000  making all strike action illegal [in C]
      B  ░░░░░░░░░░░░░░░░░░░░ 0.000  imposing restrictions on picketing [in C]
      C  ░░░░░░░░░░░░░░░░░░░░ 0.000  banning political strikes that were not primarily conce... [in C]
      D  ████████████████████ 1.000  making secondary action (in support of workers elsewher... [in C]
```
  Conformal set: {A, B, C, D}  |  Trust T = 1/|C| = 0.250

**Agent 2 (🟩 clean)**
```
      A✓ ████████████████████ 1.000  making all strike action illegal [in C]
      B  ░░░░░░░░░░░░░░░░░░░░ 0.000  imposing restrictions on picketing
      C  ░░░░░░░░░░░░░░░░░░░░ 0.000  banning political strikes that were not primarily conce...
      D  ░░░░░░░░░░░░░░░░░░░░ 0.000  making secondary action (in support of workers elsewher...
```
  Conformal set: {A}  |  Trust T = 1/|C| = 1.000

**Agent 3 (🟥 CORRUPT)**
```
      A✓ ░░░░░░░░░░░░░░░░░░░░ 0.000  making all strike action illegal [in C]
      B  ░░░░░░░░░░░░░░░░░░░░ 0.000  imposing restrictions on picketing [in C]
      C  ░░░░░░░░░░░░░░░░░░░░ 0.000  banning political strikes that were not primarily conce... [in C]
      D  ████████████████████ 1.000  making secondary action (in support of workers elsewher... [in C]
```
  Conformal set: {A, B, C, D}  |  Trust T = 1/|C| = 0.250

**Agent 4 (🟥 CORRUPT)**
```
      A✓ ░░░░░░░░░░░░░░░░░░░░ 0.000  making all strike action illegal [in C]
      B  ░░░░░░░░░░░░░░░░░░░░ 0.000  imposing restrictions on picketing [in C]
      C  ░░░░░░░░░░░░░░░░░░░░ 0.000  banning political strikes that were not primarily conce... [in C]
      D  ████████████████████ 1.000  making secondary action (in support of workers elsewher... [in C]
```
  Conformal set: {A, B, C, D}  |  Trust T = 1/|C| = 0.250

### Step 2 — Committee union (the conformal abstention signal)

⋃ᵢ Cᵢ = {A, B, C, D}  →  size **4**

⚠️  Committee disagrees → low committee confidence → **abstention candidate**

### Step 3 — CTC-Global trust-weighted score per action

`score(a) = Σᵢ πᵢ(a) · Tᵢ · 1[a ∈ Cᵢ]`

```
  A: 2.000  =  A0:1.00×1.00 + A1:0.00×0.25 + A2:1.00×1.00 + A3:0.00×0.25 + A4:0.00×0.25  ← argmax
  B: 0.000  =  A1:0.00×0.25 + A3:0.00×0.25 + A4:0.00×0.25
  C: 0.000  =  A1:0.00×0.25 + A3:0.00×0.25 + A4:0.00×0.25
  D: 0.750  =  A1:1.00×0.25 + A3:1.00×0.25 + A4:1.00×0.25
```

### Step 4 — Method comparison

| Method | Answer | Correct? |
|---|---|---|
| Majority Vote | D | ✗ |
| EntropyTrust | A | ✓ |
| CTC-Focal | A | ✓ |
| CTC-Global | A | ✓ |
| CTC-Hybrid | A | ✓ |
| Committee Conformal (selective) | A (|⋃C|=4, conc=1.00) | ✓ |

---

# Example 4

## Question (subject: sociology)

> **With industrial capitalism, leisure emerged as a concept that meant:**

**Choices**
  - **A**: distinct periods of non-work and relaxation
  - **B**: regulated and organized activities to reduce social disorder
  - **C**: the growth of civic, municipal culture through public libraries, museums and art galleries
  - **D**: all of the above ← *correct*

**Setup**: N=5 agents, k=3 corrupt under `overconfident_extreme` attack, shared q̂=1.0000

### Step 1 — Each agent emits probabilities + conformal set

**Agent 0 (🟥 CORRUPT)**
```
      A  ░░░░░░░░░░░░░░░░░░░░ 0.000  distinct periods of non-work and relaxation [in C]
      B  ░░░░░░░░░░░░░░░░░░░░ 0.000  regulated and organized activities to reduce social dis... [in C]
      C  ████████████████████ 1.000  the growth of civic, municipal culture through public l... [in C]
      D✓ ░░░░░░░░░░░░░░░░░░░░ 0.000  all of the above [in C]
```
  Conformal set: {A, B, C, D}  |  Trust T = 1/|C| = 0.250

**Agent 1 (🟩 clean)**
```
      A  ░░░░░░░░░░░░░░░░░░░░ 0.000  distinct periods of non-work and relaxation
      B  ░░░░░░░░░░░░░░░░░░░░ 0.000  regulated and organized activities to reduce social dis...
      C  ░░░░░░░░░░░░░░░░░░░░ 0.000  the growth of civic, municipal culture through public l...
      D✓ ████████████████████ 1.000  all of the above [in C]
```
  Conformal set: {D}  |  Trust T = 1/|C| = 1.000

**Agent 2 (🟥 CORRUPT)**
```
      A  ░░░░░░░░░░░░░░░░░░░░ 0.000  distinct periods of non-work and relaxation [in C]
      B  ░░░░░░░░░░░░░░░░░░░░ 0.000  regulated and organized activities to reduce social dis... [in C]
      C  ████████████████████ 1.000  the growth of civic, municipal culture through public l... [in C]
      D✓ ░░░░░░░░░░░░░░░░░░░░ 0.000  all of the above [in C]
```
  Conformal set: {A, B, C, D}  |  Trust T = 1/|C| = 0.250

**Agent 3 (🟩 clean)**
```
      A  ░░░░░░░░░░░░░░░░░░░░ 0.000  distinct periods of non-work and relaxation
      B  ░░░░░░░░░░░░░░░░░░░░ 0.000  regulated and organized activities to reduce social dis...
      C  ░░░░░░░░░░░░░░░░░░░░ 0.000  the growth of civic, municipal culture through public l...
      D✓ ████████████████████ 1.000  all of the above [in C]
```
  Conformal set: {D}  |  Trust T = 1/|C| = 1.000

**Agent 4 (🟥 CORRUPT)**
```
      A  ░░░░░░░░░░░░░░░░░░░░ 0.000  distinct periods of non-work and relaxation [in C]
      B  ████████████████████ 1.000  regulated and organized activities to reduce social dis... [in C]
      C  ░░░░░░░░░░░░░░░░░░░░ 0.000  the growth of civic, municipal culture through public l... [in C]
      D✓ ░░░░░░░░░░░░░░░░░░░░ 0.000  all of the above [in C]
```
  Conformal set: {A, B, C, D}  |  Trust T = 1/|C| = 0.250

### Step 2 — Committee union (the conformal abstention signal)

⋃ᵢ Cᵢ = {A, B, C, D}  →  size **4**

⚠️  Committee disagrees → low committee confidence → **abstention candidate**

### Step 3 — CTC-Global trust-weighted score per action

`score(a) = Σᵢ πᵢ(a) · Tᵢ · 1[a ∈ Cᵢ]`

```
  A: 0.000  =  A0:0.00×0.25 + A2:0.00×0.25 + A4:0.00×0.25  ← argmax
  B: 0.250  =  A0:0.00×0.25 + A2:0.00×0.25 + A4:1.00×0.25  ← argmax
  C: 0.500  =  A0:1.00×0.25 + A2:1.00×0.25 + A4:0.00×0.25  ← argmax
  D: 2.000  =  A0:0.00×0.25 + A1:1.00×1.00 + A2:0.00×0.25 + A3:1.00×1.00 + A4:0.00×0.25  ← argmax
```

### Step 4 — Method comparison

| Method | Answer | Correct? |
|---|---|---|
| Majority Vote | C | ✗ |
| EntropyTrust | D | ✓ |
| CTC-Focal | D | ✓ |
| CTC-Global | D | ✓ |
| CTC-Hybrid | D | ✓ |
| Committee Conformal (selective) | D (|⋃C|=4, conc=1.00) | ✓ |

---
