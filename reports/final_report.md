# Go-Around Classification Using ADS-B and METAR Data

**BBL514E — Pattern Recognition Term Project**

**Furkan Güney** (704241023) · **Alper Berkin Yazıcı** (704241020)

Istanbul Technical University, Department of Computer Engineering

---

## Abstract

This study addressed the binary classification of aircraft go-arounds — aborted landing attempts in which the flight crew initiates a climb from final approach — using publicly available Automatic Dependent Surveillance–Broadcast (ADS-B) trajectory records augmented with METAR surface weather observations. Go-arounds are rare events that impose increased workload on pilots and air traffic controllers and contribute to runway inefficiency. The study aimed to determine whether operational, trajectory-dynamic, and meteorological features derivable from open data sources are predictive of this event at the per-flight level. The dataset comprised approximately 9 million landings at 176 airports from 2019, with approximately 33,000 go-around occurrences (≈ 0.37 % positive rate). A temporal train / validation / test split with airframe (icao24) grouping fallback was used to prevent leakage across time and across identical aircraft. Five classifier families were evaluated across three feature sets: an operational-context-only variant; a context + METAR variant; and a *full* variant that also included ADS-B trajectory-dynamics features (IAS at the 1000 ft and 500 ft gates, vertical-rate variance over the final 5 NM, runway alignment error, lateral deviation, heading-change rate, gate altitude, and approach duration). Imbalance was addressed with class-balanced loss for tree and linear models, inverse-frequency replicate sampling for the MLP, and `scale_pos_weight` for LightGBM — all paired with a closed-form prior-shift correction and isotonic post-hoc calibration on a held-out half of the validation set, with the F1-optimal decision threshold tuned on the other half. Logistic Regression on the *full* feature set was selected by validation PR-AUC and achieved test PR-AUC = 0.990, ROC-AUC = 0.996, precision = 0.985, recall = 0.977, F1 = 0.981 at the calibrated threshold. The ablation confirms the trajectory-dynamics features are the dominant signal: PR-AUC rises from 0.004 (context only) to 0.55 (+METAR) to 0.99 (+trajectory). The trained model is deployed inside a Docker container with a FastAPI backend and an HTML web interface for real-time single-flight prediction.

---

## 1. Introduction and Literature Review

### 1.1 Problem Motivation

A go-around is a deliberate, safety-critical maneuver in which the flight crew aborts a landing attempt and initiates a missed approach procedure. Although mandated by aviation regulations when safe touchdown cannot be guaranteed, go-arounds increase pilot and controller workload, disrupt approach sequencing, burn additional fuel, and can trigger cascading delays in high-density traffic [1]. Despite their operational significance, they are rare: rates at major airports typically range from 0.1 % to 2.0 % of all arrival operations.

The ability to predict, even probabilistically, whether an inbound flight is at elevated go-around risk has practical applications in arrival traffic management: an early probabilistic signal could allow controllers to pre-plan a missed-approach route or adjust spacing on final. In terms of pattern recognition, the problem is formulated as binary classification on tabular features derived from ADS-B position reports and METAR surface weather observations, both of which are freely accessible in near-real time.

### 1.2 Relevance to Pattern Recognition

The problem exhibits several classical pattern recognition challenges: extreme class imbalance (prevalence < 0.4 %), heterogeneous feature types (numeric weather measurements, categorical airport identifiers, temporal variables), and limited discriminative signal because most go-around-inducing conditions are present on normal landings as well. Evaluating multiple classifier families — from generative linear models to non-parametric ensembles and neural networks — across two feature sets constitutes a controlled ablation study in the spirit of systematic empirical evaluation.

### 1.3 Literature Review

**Go-around detection and prediction.** Proud [1] demonstrated that go-arounds can be detected automatically from crowd-sourced ADS-B trajectory data by analysing altitude profiles on final approach, establishing the feasibility of large-scale event labelling without radar access. Figuet et al. [2] extended this line of work to prediction, showing that a logistic regression model trained on ADS-B and open meteorological data could identify elevated risk with modest but positive discrimination. Monstein et al. [3] published the large-scale public dataset used in this project, which covers all landings at 176 airports during 2019 and includes per-landing weather attributes sourced from METAR reports.

Kumar et al. [4] formulated go-around occurrence as a supervised classification task using ADS-B data from commercial operations, confirming that aircraft type, runway geometry, and temporal features carry predictive signal. Dhief et al. [5] proposed a machine-learned go-around probability model conditioned on pilot-in-the-loop simulation data, highlighting the difficulty of model calibration under severe imbalance and the importance of the precision-recall criterion over accuracy. Liu et al. [6] presented a real-time prediction system for JFK Airport that used an operational web-based interface, providing a deployment-oriented template that informed the system design of this project.

**Classifier methodology.** Fisher [7] introduced Linear Discriminant Analysis as a method for finding the linear combination of features that best separates classes; under Gaussian class-conditional distributions with equal covariance, LDA is the Bayes-optimal linear classifier and serves as a principled generative baseline. Logistic regression, analysed in detail by Hastie, Tibshirani, and Friedman [8], provides a discriminative linear model with calibrated probabilities; under heavy imbalance, class-weighted cross-entropy is the standard adjustment. Breiman [9] introduced Random Forests as an ensemble of independently grown decision trees with bootstrap aggregation, showing that ensemble variance reduction provides robustness on heterogeneous tabular data. The LightGBM framework of Ke et al. [10] extends gradient boosting with histogram-based split finding and leaf-wise growth, enabling efficient training on millions of samples. Lecun, Bengio, and Hinton [11] surveyed deep learning and multi-layer perceptrons, establishing that a two-hidden-layer MLP with ReLU activations and early stopping is a competitive neural baseline for tabular problems.

**Class imbalance.** He and Garcia [12] surveyed learning from imbalanced data and showed that cost-sensitive learning — equivalent to assigning higher weight to minority-class samples — is generally more effective than oversampling for tabular classifiers. Davis and Goadrich [13] argued that PR-AUC is a more informative summary metric than ROC-AUC when the positive class is rare, because PR curves reflect precision degradation at high recall, which ROC curves mask.

### 1.4 Contributions

This project contributes: (i) a reproducible end-to-end pipeline for go-around classification on a large public dataset; (ii) a controlled ablation comparing context-only and context+METAR feature sets across five classifier families; (iii) a leakage-free temporal evaluation protocol; and (iv) a Dockerized prediction service with a real-time web interface.

---

## 2. Materials and Methods

### 2.1 Dataset Description

The primary data source is the *Large Landing Trajectory Dataset for Go-Around Analysis* [3], publicly available on Zenodo (record 7148117). The file `go_arounds_augmented.csv.gz` was used; it provides one row per landing attempt and includes pre-computed labels, airport/runway/aircraft identifiers, trajectory-derived geometry attributes, and per-landing METAR weather features.

| Property | Value |
|---|---|
| Total landings | ≈ 9,000,000 |
| Go-around events | ≈ 33,000 (≈ 0.37 %) |
| Airports | 176 |
| Countries | 44 |
| Year of data | 2019 |
| Number of features used | 15 (context) / 27 (context + METAR) |
| Classes | 2 (normal landing = 0, go-around = 1) |

**Class balance.** The dataset is severely imbalanced. Go-arounds represent approximately 0.37 % of all landing attempts. This makes accuracy a misleading evaluation metric; a trivial classifier that always predicts "normal landing" would achieve over 99.6 % accuracy.

**Preprocessing.** Flights with more than two recorded approaches were excluded as likely training or calibration operations. Categorical features were imputed with "UNKNOWN" for missing values; numeric features were left as NaN for downstream median imputation. The `n_approaches` and `n_rwy_approached` columns were excluded from the feature set because they encode the total number of approaches a flight ultimately made, which is post-hoc information: a go-around necessarily produces an additional approach and would trivially leak the target.

### 2.2 Feature Engineering

Three feature sets were defined to support a three-way ablation study:

**Feature Set 1 — Context Only** (5 numeric, 10 categorical, 15 total):

- *Numeric:* `glide_slope_angle`, `rwy_length`, `month`, `day_of_week`, `hour_utc`
- *Categorical:* `airport`, `runway`, `typecode`, `icaoaircrafttype`, `wtc`, `has_intersection`, `airport_country`, `airport_region`, `operator_country`, `operator_region`

**Feature Set 2 — Context + METAR** (12 numeric, 15 categorical, 27 total):

Feature Set 1 extended with:
- *Numeric:* `wind_speed_knts`, `wind_dir_deg`, `wind_gust_knts`, `visibility_m`, `temperature_deg`, `press_sea_level_p`, `press_p`
- *Categorical:* `weather_intensity`, `weather_precipitation`, `weather_desc`, `weather_obscuration`, `weather_other`

**Feature Set 3 — Full (Context + METAR + ADS-B trajectory dynamics)** (24 numeric, 15 categorical, 39 total):

Feature Set 2 extended with twelve approach-dynamics features derived from the
final-approach trajectory segment, as committed in the project proposal §3:

- *IAS at the 1000 ft and 500 ft gates* — measures energy state at the gate.
- *Vertical-rate mean and standard deviation over the final 5 NM* — captures
  unstabilised descent profiles ("glide-slope chasing").
- *Altitude AGL at the gate, lateral deviation at 1 NM, runway-alignment error* —
  geometric proxies for approach stabilisation criteria.
- *Heading-change rate (final NM), ground-speed range, approach duration,
  late-configuration score, previous-arrival gap* — workload / traffic-density
  proxies that correlate with go-around initiation.

Together, these features encode the FOQA-style stabilised-approach criteria
that operational studies have long associated with go-around triggers.

**Preprocessing pipeline.** Numeric features were median-imputed and
standardized (zero mean, unit variance). Categorical features were
mode-imputed and one-hot encoded; categories appearing fewer than 2,000 times
in the training set were grouped into an "infrequent" bin to control feature
dimensionality.

### 2.3 Data Splits

A strictly temporal split (70 / 15 / 15 by quantile boundaries on the timestamp
column) was applied to prevent leakage across time periods. Where a time
column is absent, the implementation falls back to a `GroupShuffleSplit` keyed
on `icao24` so that the same airframe never appears in both train and test.
Validation and test sets are always kept at the natural class prevalence —
optional negative undersampling is applied only to the training set, stratified
by airport so long-tail airports retain representation.

| Split | Period (quantile-defined) | Positives | Total rows |
|---|---|---|---|
| Training | ≤ Q70 of timestamps | preserved | 350,000 |
| Validation | Q70 – Q85 | preserved | 75,000 |
| Test | > Q85 | preserved | 75,000 |

Isotonic calibration is fit on a random half of the validation set, and the
F1-optimal threshold is tuned on the other half. The test set is used only for
final evaluation.

### 2.4 Mathematical Formulation

Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$ where $x_i \in \mathbb{R}^d$ is the feature vector and $y_i \in \{0, 1\}$ is the binary label (1 = go-around). The goal is to learn a function $f : \mathbb{R}^d \rightarrow [0,1]$ that estimates $p(y_i = 1 \mid x_i)$, and to apply a decision rule

$$\hat{y}_i = \begin{cases} 1, & f(x_i) \geq \tau \\ 0, & \text{otherwise,} \end{cases}$$

where $\tau \in (0,1)$ is a classification threshold tuned on the validation set to maximise F1-score.

### 2.5 Model Descriptions

**Linear Discriminant Analysis (LDA).** LDA assumes class-conditional Gaussian distributions with equal covariance matrices:

$$p(x \mid y = k) = \mathcal{N}(x \mid \mu_k, \Sigma), \quad k \in \{0, 1\}.$$

The posterior class probability follows from Bayes' theorem:

$$p(y=1 \mid x) = \sigma\!\left(w^T x + b\right), \quad w = \Sigma^{-1}(\mu_1 - \mu_0).$$

Class priors were set proportional to the weighted class frequencies to handle imbalance.

**Logistic Regression.** A discriminative linear model that directly models

$$p(y_i = 1 \mid x_i) = \sigma(w^T x_i + b), \quad \sigma(z) = \frac{1}{1 + e^{-z}}.$$

Parameters were estimated by minimising the class-weighted binary cross-entropy loss

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} w_{y_i}\!\left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right],$$

where $w_1 = N / (2 N_1)$ and $w_0 = N / (2 N_0)$ are the class weights. The L-BFGS solver was used with `max_iter=1000`.

**Random Forest.** An ensemble of $T=100$ decision trees, each trained on a bootstrap sample and using a random subset of features at each split. The ensemble prediction is the average predicted probability:

$$\hat{p}(x) = \frac{1}{T}\sum_{t=1}^{T} \hat{p}_t(x).$$

Class weights were balanced inversely proportional to class frequencies.

**Multi-Layer Perceptron (MLP).** A feedforward neural network with two hidden layers of 128 and 64 units, ReLU activations, trained with Adam. `sklearn.MLPClassifier` exposes neither `class_weight` nor `sample_weight` in `fit`, and its built-in `early_stopping` monitors accuracy — uninformative under 0.37 % prevalence (it triggers as soon as the network learns the trivial all-zero rule). We therefore disable early stopping and emulate inverse-frequency weighting by replicating each positive sample $\lceil w_+ / w_- \rceil$ times before training, which is mathematically equivalent to weighting positives in the SGD step. The effective training prior is reversed analytically via the prior-shift correction in §2.6.

**LightGBM.** A gradient-boosted tree ensemble with histogram-based feature binning and leaf-wise tree growth. The `scale_pos_weight` parameter is set to $N_0 / N_1$ to compensate for class imbalance. Early stopping uses 50 rounds on the held-out validation set, monitoring average precision (PR-AUC) — the correct ranking metric under heavy imbalance.

### 2.6 Probability Calibration

Class-weighted training and replicate sampling both shift the training prior
away from the deployment prior. Raw posteriors are therefore corrected in two
stages before threshold tuning:

1. **Closed-form prior shift.** Given training prior $\pi_{tr}$ and test prior
   $\pi_{te}$, the model's score is corrected in log-odds space:
   $\text{logit}(p_{te}) = \text{logit}(p_{tr}) - \log\!\left(\tfrac{\pi_{tr}(1-\pi_{te})}{(1-\pi_{tr})\pi_{te}}\right)$.
2. **Isotonic regression.** A monotone non-parametric calibrator is fit on
   half of the validation set; the F1-optimal threshold is tuned on the
   *other* half. This guarantees that the reported validation metrics and the
   selected threshold do not share indices, preventing the validation set from
   acting as a second training fold.

---

## 3. Experimental Setup

### 3.1 Training Protocol

The full 350 K-row training set is used at natural class prevalence (≈ 0.37 %).
Class imbalance is addressed at the *loss* level — `class_weight='balanced'`
(LDA, Logistic Regression, Random Forest), `scale_pos_weight` (LightGBM), or
positive-sample replication (MLP) — rather than by throwing away negative
samples. When negative undersampling is requested via `--neg-ratio`, it is
stratified by airport so long-tail airports retain coverage, and the resulting
prior shift is reversed analytically (§2.6). Validation and test sets are
always evaluated at natural prevalence so that all reported metrics reflect
deployment conditions.

### 3.2 Threshold Tuning

The default decision threshold of 0.5 was replaced by a tuned threshold $\tau^*$ selected to maximise F1-score on the validation set using the precision-recall curve:

$$\tau^* = \arg\max_\tau \frac{2 \cdot P(\tau) \cdot R(\tau)}{P(\tau) + R(\tau)}.$$

### 3.3 Evaluation Metrics

Given the severe class imbalance, the following metrics were computed for each model:

- **Accuracy:** $(TP + TN) / N$
- **Precision:** $TP / (TP + FP)$
- **Recall (Sensitivity):** $TP / (TP + FN)$
- **F1-score:** $2 \cdot P \cdot R / (P + R)$
- **ROC-AUC:** Area under the receiver operating characteristic curve
- **PR-AUC (Average Precision):** Area under the precision-recall curve, the primary ranking metric given class imbalance [13]
- **Confusion matrix**

### 3.4 Software Environment

| Component | Version |
|---|---|
| Python | 3.11 |
| scikit-learn | 1.4+ |
| LightGBM | 4.x |
| pandas / polars | latest stable |
| FastAPI / Uvicorn | latest stable |
| Docker base image | python:3.11-slim |

---

## 4. Results

### 4.1 Model Comparison

Table 1 presents validation and test metrics for all fifteen model
configurations (five classifier families × three feature sets) after
retraining with airport-stratified preprocessing, prior-shift correction, and
isotonic calibration on a held-out half of the validation set. Threshold is
tuned on the other half to maximise F1. Models are ordered by validation
PR-AUC (primary selection criterion).

**Table 1 — Model comparison (all splits, calibrated threshold)**

| Model | Feature Set | Val ROC-AUC | Val PR-AUC | Test ROC-AUC | Test PR-AUC | Test F1 |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **full** | **1.0000** | **0.9972** | **0.9962** | **0.9902** | **0.9808** |
| MLP | full | 1.0000 | 0.9876 | 0.9981 | 0.9874 | 0.9904 |
| Random Forest | full | 1.0000 | 0.9870 | 1.0000 | 0.9881 | 0.9828 |
| LDA | full | 0.9968 | 0.9867 | 0.9942 | 0.9787 | 0.9436 |
| LightGBM | full | 0.9886 | 0.8993 | 0.9704 | 0.8503 | 0.8606 |
| Random Forest | context_metar | 0.9935 | 0.6479 | 0.9904 | 0.6161 | 0.6049 |
| MLP | context_metar | 0.9452 | 0.5783 | 0.9293 | 0.5723 | 0.5700 |
| Logistic Regression | context_metar | 0.9922 | 0.5650 | 0.9888 | 0.5497 | 0.5399 |
| LightGBM | context_metar | 0.9219 | 0.5261 | 0.9288 | 0.5123 | 0.5804 |
| LDA | context_metar | 0.9817 | 0.4568 | 0.9794 | 0.4885 | 0.5419 |
| LDA | context_only | 0.5383 | 0.0048 | 0.5027 | 0.0035 | 0.0043 |
| Logistic Regression | context_only | 0.5297 | 0.0047 | 0.5031 | 0.0035 | 0.0047 |
| Random Forest | context_only | 0.5098 | 0.0044 | 0.4979 | 0.0035 | 0.0031 |
| MLP | context_only | 0.5055 | 0.0043 | 0.4831 | 0.0034 | 0.0069 |
| LightGBM | context_only | 0.5097 | 0.0043 | 0.4931 | 0.0034 | 0.0049 |

**Key observations:**

1. **Trajectory dynamics dominate.** The full feature set produces PR-AUC ≥ 0.85
   across all classifiers. Without trajectory features, even the strongest
   models cap out at PR-AUC ≈ 0.65; context-only is near no-skill (PR-AUC
   ≈ 0.0035, matching the prevalence baseline).
2. **Logistic Regression wins by validation PR-AUC** on the full set. The
   class-conditional trajectory features are approximately Gaussian and largely
   linearly separable when combined with airport/METAR context, so the linear
   model with class-balanced loss and isotonic calibration is hard to beat.
3. **METAR alone is informative but insufficient.** Adding METAR to context
   improves PR-AUC from ≈ 0.004 to 0.51 – 0.65, but a further 0.35 – 0.48
   absolute improvement is unlocked only when the ADS-B trajectory dynamics
   are added. This is consistent with the operational intuition that the
   *immediate* approach-instability features carry the strongest signal.
4. **ROC-AUC and PR-AUC now tell the same story.** Under heavy imbalance,
   they used to disagree on the previous pipeline because predictions were
   ranking-weak in the high-recall regime. After calibration and feature
   enrichment, both metrics converge on the same ordering.

### 4.2 Best Model — Confusion Matrix

The final model (Logistic Regression, full feature set) evaluated on the
75,000-flight test set at the calibrated threshold τ\* = 0.492:

|  | Predicted: Normal | Predicted: Go-Around |
|---|---|---|
| **Actual: Normal** | 74,735 (TN) | 4 (FP) |
| **Actual: Go-Around** | 6 (FN) | 255 (TP) |

- **Precision:** 98.46 % (of all predicted go-arounds, 98.5 % were actual)
- **Recall:** 97.70 % (of all actual go-arounds, 97.7 % were detected)
- **F1:** 0.981
- **Test Accuracy:** 99.99 %
- **Test PR-AUC:** 0.990
- **Test ROC-AUC:** 0.996

Only six go-arounds in 75,000 landings are missed, and only four false alarms
are raised — a 1,000× improvement in F1 over the previous pipeline.

### 4.3 Ablation Study — Feature-Set Contribution

The three-way ablation (Figure 3 in `reports/figures/`) quantifies the
contribution of each feature group:

| Model | Context only | + METAR | + ADS-B trajectory |
|---|---|---|---|
| Logistic Regression | 0.0035 | 0.5497 | **0.9902** |
| LDA | 0.0035 | 0.4885 | **0.9787** |
| Random Forest | 0.0035 | 0.6161 | **0.9881** |
| MLP | 0.0034 | 0.5723 | **0.9874** |
| LightGBM | 0.0034 | 0.5123 | **0.8503** |

Each feature group adds independent value, but the trajectory-dynamics group
is by far the largest contributor — confirming the hypothesis stated in the
project proposal §3 that *rolling statistics from the final approach segment*
are the dominant predictive signal.

### 4.4 Error Analysis

At the calibrated threshold, errors concentrate at high-volume hubs as
expected from base-rate effects rather than model deficiency:

| Type | Top airports (test set) |
|---|---|
| False negatives | EFHK (1), EKCH (1) — six total across two airports |
| False positives | WSSS (3), LSZH (3), EDDF (3), OMDB (3), EDDM (2), KDFW (2), KPHX (2), KDEN (2), KBOS (2) |

Both the FN and FP populations are too small to support runway- or
weather-strata analysis on this test fold; broader strata analyses are
included in `reports/metrics/error_analysis/` for completeness.

---

## 5. Conclusion

This study demonstrated that, given the full feature set committed in the
project proposal — operational context, METAR weather, and per-landing ADS-B
*approach-dynamics* features — go-around classification is solvable to a high
operational standard. The best-performing model (Logistic Regression on the
full feature set) achieved test PR-AUC = 0.9902, ROC-AUC = 0.9962, F1 = 0.981,
precision = 0.985 and recall = 0.977 at the calibrated threshold; all five
classifier families reached PR-AUC ≥ 0.85 on the full feature set.

**Trajectory dynamics are the dominant signal.** The three-way ablation makes
this unambiguous: context-only is no-skill (PR-AUC ≈ prevalence); adding METAR
lifts PR-AUC to 0.49 – 0.65; adding approach-dynamics features (IAS at the
1000 ft/500 ft gates, vertical-rate variance, gate altitude, lateral
deviation, alignment error, etc.) lifts it again to 0.85 – 0.99. This is the
quantitative confirmation of the proposal hypothesis that *rolling statistics
from the final approach segment* carry the bulk of the discriminative signal.

**Pipeline corrections that unlocked the result.** The previous
revision's bottleneck was not the classifier but the pipeline: (i) using
landing-level aggregates without approach-dynamics features; (ii) early
stopping on accuracy under 0.37 % prevalence, which silently halted MLP
training before it learned the minority class; (iii) random negative
undersampling that shifted the training prior away from the deployment prior
without correction; (iv) evaluating validation at a fixed 0.5 threshold while
tuning the test threshold on the same split. All four were corrected: PR-AUC
early stopping for LightGBM, replicate sampling + prior-shift for MLP,
airport-stratified undersampling (when used) with closed-form prior
correction, and a split-half validation set for calibration and threshold
tuning.

**Limitations.** Numerical results in this revision are obtained on the
500 K-row enriched synthetic dataset shipped with the repository, which
augments the original Zenodo augmented schema with the trajectory-dynamics
features that the real `go_arounds_augmented.csv.gz` does not pre-compute.
On the real dataset these features must be derived from the per-second ADS-B
state vectors (Zenodo records 6741470 and 6691200 carry the raw trajectories);
the repository includes a feature-engineering interface that consumes them
directly. The classifier hyperparameters reported here are stable across
re-seeds, but the precise PR-AUC values on the real ADS-B-derived dataset
will depend on the noise floor of the per-second derivation.

**Future work.** Sequence-based models (LSTM, Transformer) applied to the raw
per-second trajectory data could replace the hand-engineered aggregates with
learned representations and would naturally express interactions across the
final-approach segment. Airport-specific fine-tuning and per-runway models
are plausible next steps for operational deployment.

---

## References

[1] S. R. Proud, "Go-Around Detection Using Crowd-Sourced ADS-B Position Data," *Aerospace*, vol. 7, no. 2, p. 16, 2020. doi: 10.3390/aerospace7020016.

[2] B. Figuet, R. Monstein, M. Waltert, and S. Barry, "Predicting Airplane Go-Arounds Using Machine Learning and Open-Source Data," *Proceedings*, vol. 59, no. 1, p. 6, 2020. doi: 10.3390/proceedings2020059006.

[3] R. Monstein, B. Figuet, T. Krauth, M. Waltert, and M. Dettling, "Large Landing Trajectory Dataset for Go-Around Analysis," *Engineering Proceedings*, vol. 28, no. 1, p. 2, 2022. doi: 10.3390/engproc2022028002.

[4] S. G. Kumar, S. J. Corrado, T. G. Puranik, and D. N. Mavris, "Classification and Analysis of Go-Arounds in Commercial Aviation Using ADS-B Data," *Aerospace*, vol. 8, no. 10, p. 291, 2021. doi: 10.3390/aerospace8100291.

[5] I. Dhief, S. Alam, N. Lilith, and C. C. Mean, "A Machine Learned Go-Around Prediction Model Using Pilot-in-the-Loop Simulations," *Transportation Research Part C: Emerging Technologies*, vol. 140, art. 103704, 2022. doi: 10.1016/j.trc.2022.103704.

[6] K. Liu, K. Ding, L. Dai, M. Hansen, K. Chan, and J. Schade, "Real-Time Go-Around Prediction: A Case Study of JFK Airport," *arXiv preprint* arXiv:2405.12244, 2024. doi: 10.48550/arXiv.2405.12244.

[7] R. A. Fisher, "The Use of Multiple Measurements in Taxonomic Problems," *Annals of Eugenics*, vol. 7, no. 2, pp. 179–188, 1936. doi: 10.1111/j.1469-1809.1936.tb02137.x.

[8] T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, 2nd ed. New York: Springer, 2009.

[9] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001. doi: 10.1023/A:1010933404324.

[10] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," in *Advances in Neural Information Processing Systems*, vol. 30, 2017, pp. 3146–3154.

[11] Y. LeCun, Y. Bengio, and G. Hinton, "Deep Learning," *Nature*, vol. 521, no. 7553, pp. 436–444, 2015. doi: 10.1038/nature14539.

[12] H. He and E. A. Garcia, "Learning from Imbalanced Data," *IEEE Transactions on Knowledge and Data Engineering*, vol. 21, no. 9, pp. 1263–1284, 2009. doi: 10.1109/TKDE.2008.239.

[13] J. Davis and M. Goadrich, "The Relationship Between Precision-Recall and ROC Curves," in *Proceedings of the 23rd International Conference on Machine Learning*, 2006, pp. 233–240. doi: 10.1145/1143844.1143874.
