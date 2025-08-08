### 1  Snapshot of the CFP (key facts you must respect)

| Item                              | Detail                                                                                                                                                                                                               | Source           |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| Research‑Topic title              | **“Causal AI: Integrating Causality and Machine Learning for Robust Intelligent Systems”**                                                                                                                           | ([Frontiers][1]) |
| Host journal / section            | *Frontiers in Artificial Intelligence* – **Logic & Reasoning in AI** (cross‑listed with *Frontiers in Big Data*)                                                                                                     | ([Frontiers][1]) |
| Mandatory themes (non‑exhaustive) | Causal representation learning; treatment‑effect estimation; causal fairness & explanations; ML for causal discovery/inference; causal RL; causal reasoning in foundation & generative models; benchmarks & datasets | ([Frontiers][1]) |
| Deadlines                         | **Manuscript‑summary** due 15 Nov 2025; **Full paper** due 27 Feb 2026                                                                                                                                               | ([Frontiers][1]) |
| Openness requirements             | Code (and, if possible, data) must be released                                                                                                                                                                       | ([Frontiers][1]) |

### 2  Where the cutting edge currently is – and the gap we can fill

| Frontier that is already crowded                          | Recent exemplars                                                                                                                                 | Unresolved pain‑points                                                                                                  |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Causal *representation* learning inside foundation models | “Causally‑Informed Pre‑training for Multimodal FMs” (arXiv 2407.19660) ([arXiv][2]) ; “Causal Foundation Models” (arXiv 2507.05333) ([arXiv][3]) | Works largely assume **static** data; they do **not** address continual updates or post‑deployment drift.               |
| Model‑level causal interpretability / editing             | “Sparse Feature Circuits” (arXiv 2403.19647) ([arXiv][4])                                                                                        | Editing is performed *once‑off*; no unified scheme to **protect** or **update** causal circuitry over time.             |
| Continual learning for FMs                                | “Future of Continual Learning in the FM Era” (arXiv 2506.03320) ([arXiv][5])                                                                     | Community treats **forgetting** and **causality** separately; catastrophic forgetting of causal mechanisms is unsolved. |
| Causal world‑models for embodied agents                   | “Essential Role of Causality in Foundation World Models” (arXiv 2402.06665) ([arXiv][6])                                                         | Again, no mechanism for *lifelong* causal adaptation under real‑world interventions.                                    |

> **Gap**: A principled framework that *continually* refreshes a foundation model’s knowledge **without eroding its learned causal mechanisms**, while making each update *auditable* through explicit interventions.

### 3  Proposed research topic

**“Intervention‑Aware Continual Causal Pre‑training (ICCP): Lifelong, Auditable Updating of Foundation Models via Structural Causal Memory”**

| Aspect                               | One‑sentence overview                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Core question**                    | *How can a deployed foundation model be updated many times (new data, regulations, safety fixes) without forgetting — and with traceable causal guarantees?*                                                                                                                                                                                                                                                                               |
| **Key idea**                         | Combine structural causal models (SCMs) with a continual‑learning controller that decides **when, where, and how** to apply an update as an *explicit intervention* on the model’s latent causal graph.                                                                                                                                                                                                                                    |
| **Novelty vs. state of the art**     | 1️⃣ Treat each incremental dataset or policy change as a *do‑operation* on the SCM; 2️⃣ Introduce **Causal Parameter Importance (CPI)** scores that protect parameters linked to validated causal pathways; 3️⃣ Use **Counterfactual Replay Buffers** that regenerate past contexts only for the *causal parent variables* at risk of being overwritten. No existing continual‑learning or FM‑editing work unifies these three components. |
| **Target article type**              | *Original Research* (methodology + empirical evaluation), plus an open‑source framework.                                                                                                                                                                                                                                                                                                                                                   |
| **Alignment with CFP bullet‑points** | ✓ Causal representation learning; ✓ Scalable causal inference; ✓ Causal reasoning in foundation models; ✓ New benchmark & evaluation metrics.                                                                                                                                                                                                                                                                                              |

#### 3.1  Methodological sketch (for the paper)

1. **Causal Representation Factorisation (CaReF)**
   *Learn a low‑rank decomposition that explicitly separates*
   (a) *stable causal mechanisms* and
   (b) *environment‑specific noise/contexts*
   during initial FM pre‑training.
   • Implementation: multi‑task contrastive loss encouraging d‑separation in latents.

2. **Structural Causal Memory (SCM‑Mem)**
   *A lightweight key–value store whose keys are learned causal variables and whose values are adapters (LoRA‑style) trained on successive data batches.*
   • Update = write a *new* adapter; inference = route queries through a learned gating network that picks the right causal branch.

3. **Causal Parameter Importance (CPI)**
   Compute importance via influence‑functions on the SCM graph; freeze or elastic‑regularise parameters whose CPI exceeds a threshold *per causal edge*.
   • Extends EWC but grounded in explicit causal structure.

4. **Counterfactual Replay Buffer (CRB)**
   Instead of raw past data, store *SCM‑encoded summaries* (parent‑variable distributions). When updating, generate counterfactual tuples to test whether new parameters would have altered validated causal pathways; if so, trigger regularisation or adapter split.

5. **Evaluation suite**

   * **Continual‑Shift‑Causal (CSC‑25)** benchmark: 25 sequential shifts (synthetic → real; 2023→2025 news; policy changes in healthcare billing).
   * Metrics: (i) **Causal Consistency Score** (retained effect estimates across tasks), (ii) **Forgetting‑Adjusted Treatment‑Effect Error**, (iii) **Auditable Update Cost**.

6. **Use‑case demos**

   * **Clinical forecasting** (new treatment guidelines release).
   * **Climate‑impact language model** that ingests monthly IPCC scenario updates.
   * **Regulatory compliance chatbot** tracking evolving data‑protection laws.

#### 3.2  Why it matters

* **Scientific payoff** – Bridges two currently siloed areas: causal inference and lifelong learning, yielding models that remain *trustworthy* under perpetual change.
* **Practical payoff** – Directly addresses industry pain‑points (e.g., LLM “model rot” and compliance updates).
* **Reproducibility & adoption** – All code, CPI visualiser, and CSC‑25 benchmark released under Apache‑2.0 to meet the CFP’s open‑science mandate.

### 4  Road‑map to submission

| Phase                          | Dates (2025–26)    | Milestones                                                                          |
| ------------------------------ | ------------------ | ----------------------------------------------------------------------------------- |
| **0. Idea registration**       | Sept – Oct 2025    | Draft 1‑page summary & secure *“manuscript summary”* slot before 15 Nov 2025.       |
| **1. Prototype & pilot data**  | Oct – Dec 2025     | Build CaReF + SCM‑Mem on open‑source Llama‑2‑7B; run 3‑shift pilot to validate CPI. |
| **2. Full experiments**        | Jan – mid‑Feb 2026 | Scale to 13B + multimodal FM; finalise CSC‑25 benchmark; ablations.                 |
| **3. Writing & artefact prep** | Feb 2026           | Paper (≤12k words), code‑repo, benchmark datasets; internal reproducibility check.  |
| **4. Submission**              | **≤27 Feb 2026**   | Upload to Frontiers portal; register GitHub DOI and CRB dataset on Zenodo.          |
| **5. Open‑review iteration**   | Mar – Jul 2026     | Address reviewer requests; release tutorial notebook.                               |

### 5  Contingency: alternate narrow‑scope spin‑offs (if resources tighten)

1. **CPI‑only paper** – Ablating the full framework, submit a *Methods* article focused on causal‑aware regularisation.
2. **Benchmark‑only paper** – Release CSC‑25 plus baselines; qualifies as *Data Report* or *Technology & Code*.
3. **Domain‑specific demo** – “Causal Continual Survival Analysis for Oncology” as a *Brief Research Report*.

---

#### Bottom line

By explicitly **treating every model update as a causal intervention** and safeguarding the discovered mechanisms throughout a model’s lifetime, the proposed ICCP framework squarely targets the robustness, generalisability and explainability goals articulated in the CFP. No published work yet delivers a unified, reproducible solution to **continual causal adaptation in foundation models**—making this a strong, novel candidate for the *Frontiers in AI* Research Topic.

[1]: https://www.frontiersin.org/research-topics/73155/causal-ai-integrating-causality-and-machine-learning-for-robust-intelligent-systems "Frontiers | Causal AI: Integrating Causality and Machine Learning for Robust Intelligent Systems"
[2]: https://arxiv.org/pdf/2407.19660v3?utm_source=chatgpt.com "A Causally Informed Pretraining Approach for Multimodal Foundation ..."
[3]: https://arxiv.org/abs/2507.05333?utm_source=chatgpt.com "[2507.05333] Causal Foundation Models: Disentangling Physics from ..."
[4]: https://arxiv.org/abs/2403.19647?utm_source=chatgpt.com "Sparse Feature Circuits: Discovering and Editing Interpretable Causal ..."
[5]: https://arxiv.org/abs/2506.03320?utm_source=chatgpt.com "The Future of Continual Learning in the Era of Foundation Models: Three ..."
[6]: https://arxiv.org/abs/2402.06665?utm_source=chatgpt.com "The Essential Role of Causality in Foundation World Models for Embodied AI"

