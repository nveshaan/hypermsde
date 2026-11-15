# MSDE Zero-Shot and Few-Shot Anomaly Detection Benchmark Plan

# 1. Central Research Question

> **Can the geometry of pretrained medical-image embeddings be exploited for zero-shot anomaly detection in a completely unsupervised and training-free way?**

The paper investigates whether pretrained medical-image representations contain sufficiently structured geometric information to distinguish anomalous from normal samples **without anomaly labels, target-task training, or target support examples**.

The core hypothesis is:

> **Pretrained medical-image embeddings contain useful geometric structure, and MSDE can exploit this structure directly to perform zero-shot anomaly detection.**

The experimental story therefore separates three factors:

1. **Representation:** Does the pretrained embedding space contain useful anomaly information?
2. **Detector:** Does MSDE extract more of that information than standard geometric/text-guided detectors?
3. **Interaction:** Does MSDE consistently benefit from different pretrained medical representations?

Few-shot experiments then ask how the same approach behaves when a small amount of normal target information is added.

---

# 2. Core Methodological Positioning

MSDE is evaluated as:

- **Training-free:** no target-task model parameters are optimized.
- **Unsupervised:** no target anomaly labels are used.
- **Zero-shot:** no target support examples are required.
- **Transductive:** the target/query set can be used collectively.
- **Few-shot:** optionally extended using a small normal-only support set.

MSDE is evaluated in two feature settings:

- **MSDE (CLIP):** OpenAI CLIP.
- **MSDE (Best4):** the best-performing encoder among MedImageInsight, UniMed-CLIP, BiomedCLIP, and MedSigLIP.

The key distinction is that **density itself is not the anomaly score**. MSDE uses the geometry/dynamics of density enhancement, with anomaly scores based on the resulting movement/displacement of samples.

---

# 3. Terminology

| Term | Meaning in this benchmark |
|---|---|
| **Zero-shot (Z)** | No target support examples or target labels are supplied. |
| **Few-shot (F)** | A small number of target **normal** support examples are supplied. |
| **Unsupervised (U)** | No target anomaly labels are used. |
| **Training-free** | No target-task model parameters are optimized. |
| **Transductive (T)** | The method can use the target/query set collectively. |
| **Inductive (I)** | The method can score a query using a fixed reference/support representation without jointly processing the query set. |

## 3.1 Property overview

| Method | Training | Supervision | Regime | Inference |
|---|---|---|---|---|
| **AnomalyCLIP** | Auxiliary training | Auxiliary anomaly information | Zero-shot | Inductive |
| **AdaCLIP** | Auxiliary training | Auxiliary anomaly information | Zero-shot | Inductive |
| **FAPrompt** | Auxiliary training | Auxiliary training data | Zero-shot | Inductive |
| **AA-CLIP** | Auxiliary training | Auxiliary training data | Zero-shot | Inductive |
| **Bayes-PFL** | Auxiliary training | Auxiliary training data | Zero-shot | Inductive |
| **MSDE** | **None** | **None** | **Zero-shot** | **Transductive** |
| **ANoCo** | None | None | Few-shot — normal support | Inductive |
| **DPMM** | Support-set fitting | None | Few-shot — normal support | Inductive |
| **MediCLIP** | Few-shot adaptation | Synthetic anomalies used for adaptation | Few-shot — normal support | Inductive |
| **UniVAD** | None | None | Few-shot — normal support | Inductive |
| **AnomaMed** | Few-shot adaptation | Synthetic anomalies used for adaptation | Few-shot — normal support | Inductive |
| **MSDE** | **None** | **Normal support only** | **Few-shot — normal support** | **Transductive/Inductive** |

---

# 4. Main Experimental Story

The main paper should answer the following questions in sequence:

### Question 1 — Can pretrained medical embeddings support zero-shot anomaly detection?

→ **Table 1: Zero-Shot Benchmark**

### Question 2 — Is the gain actually coming from MSDE rather than simply from the pretrained encoder?

→ **Table 2: Encoder × Detector**

### Question 3 — Does MSDE exploit useful structure across different pretrained representations?

→ **Table 3: MSDE × Encoder**

### Question 4 — What happens when a small amount of normal target information is added?

→ **Figure 1: Data Efficiency Curve**

### Question 5 — Which components of the proposed MSDE formulation are responsible for its improvement?

→ **Table 4: Modifications over Original MSDE**

### Question 6 — How much of the performance comes from exploiting the unlabeled target distribution?

→ **Table 5: Transductive vs Inductive**

This ordering is intended to make the paper read as:

> **Embedding geometry exists → MSDE can exploit it → the gain is not merely the encoder → the effect generalizes across encoders → MSDE is data-efficient → the geometric modifications explain the gain → transductive access to the target distribution provides additional information.**

---

# 5. Zero-Shot Benchmark

## 5.1 Research question

> **Can pretrained medical-image embeddings support anomaly detection without target support, anomaly labels, or target-task training?**

## 5.2 Protocol

- No target support images are supplied.
- No target anomaly labels are supplied.
- MSDE (CLIP) uses OpenAI CLIP ViT-L/14@336 image embeddings, the same CLIP weights as the published baselines (which use its patch tokens).
- MSDE (Best4) uses the best of the four medical encoders.
- Classical training-free operations may exploit the unlabeled target set collectively.
- The Best4 encoder must be selected without using final test-set anomaly labels.

## 5.3 Methods

| Method | Feature space | Target support | Target labels | Target training | Role |
|---|---|---|---|---|---|
| AnomalyCLIP | OpenAI CLIP ViT-L/14@336 patch tokens | None | None | None | Published ZSAD |
| AdaCLIP | OpenAI CLIP ViT-L/14@336 patch tokens | None | None | None | Published ZSAD |
| FAPrompt | OpenAI CLIP ViT-L/14@336 patch tokens | None | None | None | Published ZSAD |
| AA-CLIP | OpenAI CLIP ViT-L/14@336 patch tokens (to verify) | None | None | None | Published ZSAD |
| Bayes-PFL | OpenAI CLIP ViT-L/14@336 patch tokens (to verify) | None | None | None | Published ZSAD |
| **MSDE (CLIP)** | **OpenAI CLIP ViT-L/14@336 image embedding** | **None** | **None** | **None** | **Proposed** |
| **MSDE (Best4)** | **Best of MedImageInsight / UniMed-CLIP / BiomedCLIP / MedSigLIP** | **None** | **None** | **None** | **Proposed** |

## 5.4 Zero-shot does not imply unsupervised

Zero-shot does not imply unsupervised.

A method can be zero-shot but use supervised or auxiliary training performed before deployment, as long as it requires no target-domain support examples at inference.

- **AnomalyCLIP:** zero-shot, but trained using auxiliary data.
- **AdaCLIP:** zero-shot, but trained using auxiliary data.
- **FAPrompt:** zero-shot, but trained using auxiliary data.
- **AA-CLIP:** zero-shot, but trained using auxiliary data.
- **Bayes-PFL:** zero-shot, but trained using auxiliary data.
- **MSDE:** zero-shot, unsupervised, training-free.

---

# 6. Encoder × Detector Comparison

## 6.1 Research question

> **Is MSDE actually extracting useful information from the embedding geometry, or is its performance primarily due to the quality of the pretrained encoder?**

This is the key controlled experiment.

The representation is held fixed while the detector changes.

## 6.2 Encoders

- OpenAI CLIP
- MedImageInsight
- UniMed-CLIP
- BiomedCLIP
- MedSigLIP

## 6.3 Detectors

- Cosine softmax with text embedding
- kNN
- LOF
- PCA reconstruction
- MSDE

## 6.4 Encoder × Detector Matrix

| Encoder | Cosine softmax + text embedding | kNN | LOF | PCA | **MSDE** |
|---|---:|---:|---:|---:|---:|
| **OpenAI CLIP** | | | | | |
| **MedImageInsight** | | | | | |
| **UniMed-CLIP** | | | | | |
| **BiomedCLIP** | | | | | |
| **MedSigLIP** | | | | | |

### Interpretation

This table should establish whether MSDE provides an advantage **within the same representation**.

For example, if:

> MSDE > kNN, LOF, PCA, and text-guided scoring

across multiple encoders, the improvement cannot simply be attributed to using a better pretrained representation.

This directly addresses:

> **Does MSDE contribute beyond the learned encoder?**

---

# 7. MSDE × Encoder Comparison

## 7.1 Research question

> **Does MSDE exploit useful geometric structure across different pretrained medical representations?**

## 7.2 Protocol

- Same dataset split for every encoder.
- Same MSDE configuration.
- Same evaluation protocol.
- No test labels used to select the encoder.
- Report MSDE separately with each encoder.
- Best4 is reported as a practical upper-end operating point, but does not replace the per-encoder results.

## 7.3 Main MSDE × Encoder Table

| Encoder | AUROC | AUPRO |
|---|---:|---:|
| OpenAI CLIP | | |
| MedImageInsight | | |
| UniMed-CLIP | | |
| BiomedCLIP | | |
| MedSigLIP | | |
| **Best4** | | |

The purpose is not simply to find the "best encoder."

The important question is whether **MSDE remains effective across substantially different pretrained embedding spaces**.

---

# 8. Few-Shot Extension

## 8.1 Research question

> **If MSDE can detect anomalies without target examples, how does its performance change when a small number of normal target examples are provided?**

Few-shot is therefore presented as an **extension of the zero-shot result**, rather than as a separate primary motivation.

## 8.2 Protocol

The main few-shot benchmark is **normal-only**.

For a given shot count `K`:

> Provide exactly **K normal support images** to each method and evaluate on the held-out test set.

No abnormal support images or anomaly labels are provided.

Recommended:

**K ∈ {1, 2, 4, 8, 16}**

Zero-shot can be viewed conceptually as **K = 0**, although its protocol differs because the target set may be used transductively.

## 8.3 Support-set rules

- The support set comes from the designated training/support split.
- The test set remains completely separate.
- All normal-only methods receive the same number `K` of normal support images.
- Methods do not bring their own private support set.
- Support selection follows a fixed/randomized protocol shared across methods.
- No anomaly labels are used to construct or select the support set.

## 8.4 Few-shot methods

| Method | Feature space | Support | Target adaptation/training | Real abnormal support? | Role |
|---|---|---|---|---:|---|
| ANoCo | DINOv3-L/16 patch features (layer 18) | K normal images | None | No | Published training-free FSAD |
| DPMM | DINOv2-S/14 patch features | K normal images | Fit mixture model | No | Published unsupervised FSAD |
| MediCLIP | CLIP patch features (variant to verify) | K normal images | Published adaptation | No | Published medical FSAD |
| UniVAD | CLIP and DINOv2 patch features with segmentation masks (to verify) | K normal images | None | No | Published training-free FSAD |
| AnomaMed | Multi-layer CLIP patch features (variant to verify) | K normal images | Synthetic anomaly generation + adaptation | No | Published medical FSAD |
| **MSDE (CLIP)** | **OpenAI CLIP ViT-L/14@336 image embedding** | **K normal images** | **None** | **No** | **Training-free FSAD** |
| **MSDE (Best4)** | **Best of MedImageInsight / UniMed-CLIP / BiomedCLIP / MedSigLIP** | **K normal images** | **None** | **No** | **Training-free FSAD** |

---

# 9. Data Efficiency

## 9.1 Research question

> **How much target-domain information does MSDE actually need?**

Evaluate:

**K ∈ {0, 1, 2, 4, 8, 16}**

where:

- `K = 0`: zero-shot
- `K > 0`: few-shot normal support

## 9.2 Figure 1 — Data Efficiency Curve

- **x-axis:** K ∈ {0, 1, 2, 4, 8, 16}
- **y-axis:** AUROC and AUPRO where applicable
- **Panels:** Chest, Fundus, MRI, OCT, plus optional mean panel
- **Lines:** MSDE (CLIP), MSDE (Best4), few-shot baselines at K ≥ 1
- **Reference:** best zero-shot baseline
- **Uncertainty:** mean ± standard deviation over support-sampling seeds
- Same held-out test set for every K

The desired story is:

> **Strong performance at K = 0 demonstrates zero-shot capability; improvement with increasing K demonstrates that MSDE can additionally exploit small amounts of normal target information.**

---

# 10. Modifications over Original MSDE

## 10.1 Research question

> **Why does the proposed MSDE formulation work better than the original MSDE?**

Start from the original MSDE of Kar et al. (arXiv:2602.03293) and add the repo's modifications one at a time.

Each row keeps every modification above it.

## 10.2 Cumulative ablation

| Step | Modification | Overrides relative to row above |
|---|---|---|
| 0 | **Original MSDE**: raw features, plain kNN reselected every iteration, paper density, weights and score, paper hyperparameters | `density=original score=original weight=original metric=original model.sphere=0 model.recompute_neighbors=1` |
| 1 | + Fixed neighbours | `model.recompute_neighbors=0` |
| 2 | + Unit-sphere normalization | `model.sphere=1` |
| 3 | + Fréchet-mean centring | `model.sphere=2` |
| 4 | + Proposed density | `density=default` |
| 5 | + Proposed neighbour score | `score=default` |
| 6 | + Proposed neighbour weights | `weight=default` |
| 7 | + Proposed anomaly metric (= default MSDE) | `metric=default` |

## 10.3 Main ablation table

**AUROC (%), zero-shot, OpenAI CLIP**

| Step | Configuration | Chest | Fundus | MRI | OCT | Mean | Δ prev | Δ orig |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | Original MSDE | | | | | | — | — |
| 1 | + Fixed neighbours | | | | | | | |
| 2 | + Unit sphere | | | | | | | |
| 3 | + Fréchet-mean centring | | | | | | | |
| 4 | + Proposed density | | | | | | | |
| 5 | + Proposed neighbour score | | | | | | | |
| 6 | + Proposed neighbour weights | | | | | | | |
| 7 | + Proposed anomaly metric (**default MSDE**) | | | | | | | |

Complement the cumulative ablation with a **leave-one-out ablation** from the final configuration.

---

# 11. Transductive vs Inductive MSDE

## 11.1 Research question

> **Does exploiting the unlabeled target distribution contribute to anomaly detection?**

Compare:

- **Transductive MSDE:** uses the target/query set collectively.
- **Inductive MSDE:** uses only a fixed normal support/reference set when constructing the anomaly score.

## 11.2 Table

| Variant | Uses target/query set collectively? | Mean AUROC | Mean AUPRO |
|---|---:|---:|---:|
| MSDE — inductive | No | | |
| **MSDE — transductive** | **Yes** | | |
| **Δ** | | | |

This isolates the information obtained from the **unlabeled target distribution itself**.

---

# 12. Main Paper Tables

The main paper should use the following order.

## Table 1 — Zero-Shot Benchmark

**Question: Can pretrained medical-image embeddings support completely unsupervised and training-free ZSAD?**

| Method | Encoder | Target support | Training-free? | Unsupervised? | Transductive? | Mean AUROC |
|---|---|---|---:|---:|---:|---:|
| AnomalyCLIP | CLIP | None | No | No | No | |
| AdaCLIP | CLIP | None | No | No | No | |
| FAPrompt | CLIP | None | No | No | No | |
| AA-CLIP | CLIP | None | No | No | No | |
| Bayes-PFL | CLIP | None | No | No | No | |
| **MSDE (CLIP)** | **CLIP** | **None** | **Yes** | **Yes** | **Yes** | |
| **MSDE (Best4)** | **Best4 medical encoder** | **None** | **Yes** | **Yes** | **Yes** | |

Per-modality AUROC/AUPRO should be included in the full table if space permits.

---

# Table 2 — Encoder × Detector

**Question: Does MSDE provide an advantage when the pretrained representation is held fixed?**

| Encoder | Cosine softmax + text embedding | kNN | LOF | PCA | **MSDE** |
|---|---:|---:|---:|---:|---:|
| OpenAI CLIP | | | | | |
| MedImageInsight | | | | | |
| UniMed-CLIP | | | | | |
| BiomedCLIP | | | | | |
| MedSigLIP | | | | | |

This is the principal **representation-vs-detector disentanglement experiment**.

---

# Table 3 — MSDE × Encoder

**Question: Does MSDE exploit useful structure across different pretrained representations?**

| Encoder | AUROC | AUPRO |
|---|---:|---:|
| OpenAI CLIP | | |
| MedImageInsight | | |
| UniMed-CLIP | | |
| BiomedCLIP | | |
| MedSigLIP | | |
| **Best4** | | |

---

# Figure 1 — Data Efficiency

**Question: How does MSDE behave as target-domain information increases?**

Plot performance against:

**K = 0, 1, 2, 4, 8, 16**

The figure should show:

- MSDE (CLIP)
- MSDE (Best4)
- Few-shot baselines for K ≥ 1
- Best zero-shot baseline as a horizontal reference
- Mean ± standard deviation for K > 0

---

# Table 4 — Modifications over Original MSDE

**Question: Which changes to MSDE produce the improvement?**

Use the cumulative ablation from Section 10.

| Step | Configuration | Chest | Fundus | MRI | OCT | Mean | Δ prev | Δ orig |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | Original MSDE | | | | | | — | — |
| 1 | + Fixed neighbours | | | | | | | |
| 2 | + Unit sphere | | | | | | | |
| 3 | + Fréchet-mean centring | | | | | | | |
| 4 | + Proposed density | | | | | | | |
| 5 | + Proposed neighbour score | | | | | | | |
| 6 | + Proposed neighbour weights | | | | | | | |
| 7 | + Proposed anomaly metric (**default MSDE**) | | | | | | | |

---

# Table 5 — Transductive vs Inductive

**Question: Does exploiting the unlabeled target distribution itself provide useful information?**

| Variant | Target set jointly used? | Mean AUROC | Mean AUPRO |
|---|---:|---:|---:|
| MSDE — inductive | No | | |
| **MSDE — transductive** | **Yes** | | |
| **Δ** | | | |

---

# Table 6 — Encoder × K-Shot

**Question: Does the benefit of normal support depend on the pretrained representation?**

| Encoder | K = 0 | K = 1 | K = 2 | K = 4 | K = 8 | K = 16 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenAI CLIP | | | | | | | |
| MedImageInsight | | | | | | | |
| UniMed-CLIP | | | | | | | |
| BiomedCLIP | | | | | | | |
| MedSigLIP | | | | | | | |

Per-modality versions go in the appendix.

---

# 13. Appendix Results

The following should primarily support the main story rather than compete with it:

- Full per-modality zero-shot results.
- Full per-modality encoder × detector matrices.
- Full few-shot benchmark tables.
- Per-modality data-efficiency tables.
- Per-encoder K-shot results.
- Leave-one-out MSDE ablation.
- Additional MSDE configuration ablations.
- Detailed method-property table.
- Hyperparameter sensitivity.
- Computational cost.
- Feature dimensionality and preprocessing details.
- Verification details for baseline feature spaces/configurations.

---

# 14. Fairness and Interpretation

## 14.1 Method-level comparison

**MSDE (CLIP)** vs other detectors using the same OpenAI CLIP representation.

Question:

> **Does MSDE provide an advantage over standard text-guided, geometric, density, and subspace anomaly detectors when the representation is held fixed?**

---

## 14.2 Representation-level comparison

**MSDE** across five pretrained encoders.

Question:

> **How much does the choice of pretrained representation affect MSDE?**

---

## 14.3 Representation–detector interaction

The encoder × detector experiment separates:

\[
\text{representation quality}
\]

from

\[
\text{detector quality}.
\]

The strongest evidence for the proposed approach is not simply that MSDE achieves high AUROC, but that:

> **MSDE consistently extracts more useful anomaly information from the same representation than conventional alternatives.**

---

## 14.4 Best4

**MSDE (Best4)** demonstrates the performance achievable when MSDE is paired with the strongest candidate medical representation.

The encoder-selection procedure must not use final test labels.

Best4 is therefore a practical operating point, not the primary scientific claim.

---

# 15. Final Experimental Story

The paper should ultimately tell the following story.

### 1. Start with the question

> **Can the geometry of pretrained medical-image embeddings be exploited for zero-shot anomaly detection without anomaly labels, target training, or target support?**

↓

### 2. Establish zero-shot feasibility

**Table 1**

MSDE performs competitively in zero-shot anomaly detection while being:

- training-free,
- unsupervised,
- target-support-free,
- transductive.

↓

### 3. Establish that MSDE itself contributes

**Table 2**

When the encoder is held fixed, MSDE outperforms conventional detectors.

Therefore the result is not simply:

> "a good medical encoder produces good anomaly scores."

Instead:

> **MSDE extracts additional anomaly-relevant information from the pretrained embedding geometry.**

↓

### 4. Establish representation generality

**Table 3**

MSDE works across multiple pretrained medical representations.

Therefore the principle is not restricted to one encoder.

↓

### 5. Establish data efficiency

**Figure 1**

MSDE starts at **K = 0** and can incorporate increasing amounts of normal target information.

This connects zero-shot and few-shot detection through a single data-efficiency curve.

↓

### 6. Explain the improvement

**Table 4**

The cumulative ablation identifies which modifications to original MSDE produce the improvement.

↓

### 7. Explain transductivity

**Table 5**

The transductive-vs-inductive experiment determines whether access to the unlabeled target distribution is itself responsible for part of the gain.

↓

### 8. Extend the representation analysis to few-shot

**Table 6**

Determine whether the effect of target support depends on the underlying pretrained representation.

---

# 16. Central Paper Positioning

The central methodological positioning should therefore be:

> **We investigate whether the geometry of pretrained medical-image embeddings can be directly exploited for zero-shot anomaly detection without anomaly labels, target-task training, or target support examples. We show that MSDE can extract anomaly-relevant structure from these representations in a training-free, unsupervised, and transductive manner, and that the approach remains effective across multiple pretrained medical encoders.**

The paper is consequently positioned around:

\[
\boxed{
\text{Pretrained representation}
\rightarrow
\text{embedding geometry}
\rightarrow
\text{MSDE}
\rightarrow
\text{zero-shot anomaly detection}
}
\]

rather than:

\[
\boxed{
\text{Existing MSDE}
\rightarrow
\text{apply it to CLIP}
}
\]

The latter makes the work sound primarily like an application/adaptation. The former makes the **scientific question about exploiting pretrained embedding geometry** the organizing principle of the paper.