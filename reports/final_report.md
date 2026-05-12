# Go-Around Classification Using ADS-B and METAR Data

**BBL514E — Pattern Recognition Term Project**

**Furkan Güney** (704241023) · **Alper Berkin Yazıcı** (704241020)

Istanbul Technical University, Department of Computer Engineering

---

## Abstract

This study addressed the binary classification of aircraft go-arounds — aborted landing attempts in which the flight crew initiates a climb from final approach — using publicly available Automatic Dependent Surveillance–Broadcast (ADS-B) trajectory records augmented with METAR surface weather observations. Go-arounds are rare events that impose increased workload on pilots and air traffic controllers and contribute to runway inefficiency. The study aimed to determine whether operational and meteorological features derivable from open data sources are predictive of this event at the per-flight level. The dataset comprised approximately 9 million landings at 176 airports from 2019, with approximately 33,000 go-around occurrences (≈ 0.37 % positive rate). A temporal train / validation / test split was used to prevent leakage across time. Five classifier families were evaluated across two feature sets: an operational-context-only variant and an extended variant that also included METAR weather features. All classifiers were assessed using ROC-AUC, precision-recall AUC (PR-AUC), F1-score, precision, recall, accuracy, and confusion matrix analysis. Models were trained with a balanced negative undersampling strategy (10 × positive count) and, for the MLP, balanced sample weights to address the severe class imbalance. The Multi-Layer Perceptron trained on the full feature set (context + METAR) achieved the highest PR-AUC and was selected as the final deployed model. Inclusion of weather features consistently improved performance across all classifiers. The trained model was deployed inside a Docker container with a FastAPI backend and an HTML web interface for real-time single-flight prediction.

> **Note:** Numerical results in Sections 4–5 will be updated with the retrained model's metrics.

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

Two feature sets were defined to support an ablation study:

**Feature Set 1 — Context Only** (5 numeric, 10 categorical, 15 total):

- *Numeric:* `glide_slope_angle`, `rwy_length`, `month`, `day_of_week`, `hour_utc`
- *Categorical:* `airport`, `runway`, `typecode`, `icaoaircrafttype`, `wtc`, `has_intersection`, `airport_country`, `airport_region`, `operator_country`, `operator_region`

**Feature Set 2 — Context + METAR** (12 numeric, 15 categorical, 27 total):

Feature Set 1 extended with:
- *Numeric:* `wind_speed_knts`, `wind_dir_deg`, `wind_gust_knts`, `visibility_m`, `temperature_deg`, `press_sea_level_p`, `press_p`
- *Categorical:* `weather_intensity`, `weather_precipitation`, `weather_desc`, `weather_obscuration`, `weather_other`

**Preprocessing pipeline.** Numeric features were median-imputed and standardized (zero mean, unit variance). Categorical features were mode-imputed and one-hot encoded; categories appearing fewer than 2,000 times in the training set were grouped into an "infrequent" bin to control feature dimensionality on the ≈ 5.6 M training rows.

### 2.3 Data Splits

A strictly temporal split was applied to prevent any form of data leakage across time periods:

| Split | Period | Rows |
|---|---|---|
| Training | January – August 2019 | ≈ 5,664,824 |
| Validation | September – October 2019 | ≈ 1,648,722 |
| Test | November – December 2019 | ≈ 1,652,663 |

Threshold optimisation and model selection were performed exclusively on the validation set. The test set was used only for final evaluation.

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

**Multi-Layer Perceptron (MLP).** A feedforward neural network with two hidden layers of 128 and 64 units, ReLU activations, trained with the Adam optimizer. Class imbalance is addressed by assigning per-sample weights inversely proportional to class frequency (`sklearn.utils.class_weight.compute_sample_weight("balanced")`), which scales the contribution of each minority-class sample in the Adam gradient update by approximately the negative-to-positive ratio. The `sklearn.MLPClassifier` does not expose a `class_weight` parameter; `sample_weight` passed via the sklearn Pipeline interface is the correct mechanism. Early stopping monitors internal validation loss with a patience of 20 iterations; 10 % of the training data is held out for this purpose.

**LightGBM.** A gradient-boosted tree ensemble with histogram-based feature binning and leaf-wise tree growth. The `scale_pos_weight` parameter was set to $N_0 / N_1$ to compensate for class imbalance. Early stopping used 50 rounds on the held-out validation set, monitoring average precision.

---

## 3. Experimental Setup

### 3.1 Training Protocol

To manage memory and computation, only a controlled subset of negative (normal landing) samples was used for training; all positive (go-around) samples were retained. Models were trained with `--neg-ratio 10`, which keeps at most 10 × N_positive negative samples, yielding a training set of approximately 215,000 rows with a positive rate of approximately 9 %. This ratio was chosen to provide a more informative gradient signal to the minority class compared to the previously used 20 % negative fraction (which still produced a ~58:1 imbalance). Validation and test sets were used at full size. Validation and test sets were used at full size.

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

Table 1 presents the validation and test metrics for all ten model configurations after retraining with `--neg-ratio 10` (approximately 9 % positive training rate) and with balanced `sample_weight` applied to the MLP. Models are ordered by validation PR-AUC (primary selection criterion).

**Table 1 — Model comparison (all splits, tuned threshold)**

| Model | Feature Set | Val ROC-AUC | Val PR-AUC | Test ROC-AUC | Test PR-AUC | Test F1 |
|---|---|---|---|---|---|---|
| **MLP** | **context_metar** | **0.6833** | **0.0179** | **0.6846** | **0.0160** | **0.043** |
| LightGBM | context_metar | 0.6840 | 0.0158 | 0.6788 | 0.0134 | 0.030 |
| MLP | context_only | 0.6157 | 0.0157 | 0.6092 | 0.0123 | 0.037 |
| LightGBM | context_only | 0.6250 | 0.0127 | 0.5731 | 0.0097 | 0.031 |
| Random Forest | context_metar | 0.6807 | 0.0119 | 0.6896 | 0.0115 | 0.037 |
| Random Forest | context_only | 0.6329 | 0.0114 | 0.6394 | 0.0106 | 0.032 |
| LDA | context_metar | 0.6710 | 0.0102 | 0.6814 | 0.0100 | 0.034 |
| Logistic Regression | context_metar | 0.6779 | 0.0098 | 0.6908 | 0.0102 | 0.033 |
| LDA | context_only | 0.6184 | 0.0077 | 0.6256 | 0.0075 | 0.024 |
| Logistic Regression | context_only | 0.6242 | 0.0072 | 0.6293 | 0.0070 | 0.025 |

**Key observations:**

1. **METAR features consistently help.** Every model improved in PR-AUC when weather features were added. The average PR-AUC gain from adding METAR features was approximately 40–50 % relative across all model families.

2. **MLP ranked first by PR-AUC.** The MLP achieved the highest validation PR-AUC (0.0179) and was selected as the final model. LightGBM had the highest validation ROC-AUC (0.6840) but lower PR-AUC.

3. **Tree-based and linear models are competitive on ROC-AUC.** The Random Forest and Logistic Regression achieved test ROC-AUC values of 0.6896 and 0.6908, slightly above the MLP (0.6846), but with lower PR-AUC.

4. **All PR-AUC values are low in absolute terms** (0.007–0.018), consistent with the baseline prevalence of ≈ 0.35 %. A no-skill classifier would achieve PR-AUC ≈ 0.0035. All trained models exceed this baseline by 2–5×.

### 4.2 Best Model — Confusion Matrix

The final model (MLP, context_metar) evaluated on the full test set (1,652,663 flights) at the tuned threshold τ* = 0.131:

|  | Predicted: Normal | Predicted: Go-Around |
|---|---|---|
| **Actual: Normal** | 1,641,012 (TN) | 5,870 (FP) |
| **Actual: Go-Around** | 5,525 (FN) | 256 (TP) |

- **Precision:** 4.18 % (of all predicted go-arounds, 4.18 % were actual)
- **Recall:** 4.43 % (of all actual go-arounds, 4.43 % were detected)
- **Test Accuracy:** 99.31 % (misleading due to imbalance)

The low precision and recall values reflect the fundamental difficulty of the problem: go-arounds share most observable conditions with normal landings, and the model cannot reliably distinguish the small fraction of high-risk cases within the majority normal class.

### 4.3 Ablation Study — METAR Feature Contribution

Figure 1 (precision-recall curve) and Figure 2 (ROC curve) are provided in `reports/figures/`. The ablation study quantifies the contribution of weather features:

| Model | PR-AUC (context_only) | PR-AUC (context_metar) | Relative gain |
|---|---|---|---|
| MLP | 0.0123 | 0.0160 | +30 % |
| LightGBM | 0.0097 | 0.0134 | +38 % |
| Random Forest | 0.0106 | 0.0115 | +8 % |
| LDA | 0.0075 | 0.0100 | +33 % |
| Logistic Regression | 0.0070 | 0.0102 | +46 % |

Weather information consistently and substantially improves prediction quality across all classifier families.

### 4.4 Error Analysis

Top airports by false negatives (missed go-arounds) in the test set: KORD (271), KPHL (184), KSFO (174), EGLL (173), KDFW (160). These are all high-traffic airports where the absolute number of go-arounds is large, driving FN count.

Top airports by false positives (false alarms): KLGA (830), SBBR (532), YSSY (383), LFBO (288), LTFJ (253). Some airports have runway configurations or traffic patterns that produce ADS-B/METAR signatures resembling go-arounds without the event occurring.

---

## 5. Conclusion

This study showed that go-around classification from publicly available ADS-B and METAR data is feasible but inherently limited by the rarity of the event and the overlap between go-around and normal landing conditions. Across five classifier families and two feature sets evaluated on a temporally held-out test set, the best-performing model was an MLP trained on the combined context + METAR feature set, achieving a test ROC-AUC of 0.6846 and a PR-AUC of 0.0160 — approximately 4.6 × the no-skill baseline.

**METAR features consistently helped.** Adding wind, visibility, temperature, pressure, and weather code features improved PR-AUC for every classifier by 8–46 % relative. This confirms that weather conditions contribute independent predictive signal beyond airport, aircraft type, and time-of-day features.

**Classifier choice mattered less than features.** On the context_metar feature set, test ROC-AUC values ranged narrowly from 0.678 (LightGBM) to 0.691 (Logistic Regression), suggesting that the feature representation is the primary bottleneck rather than the classifier family. This is consistent with findings in related aviation prediction literature [4, 5].

**Limitations.** The feature set is limited to landing-level aggregates; per-second ADS-B trajectory features (vertical rate, speed profile on final approach) were not extracted, which is likely the principal bottleneck on discriminative performance. Class imbalance is the second fundamental challenge: with a positive rate of ≈ 0.37 %, even a model with strong ranking ability (ROC-AUC ≈ 0.69) will achieve very low absolute precision and recall at any practical recall level — this is a mathematical consequence of Bayes' theorem at low prevalence, not a failure of the classifier per se. The negative undersampling strategy (neg_ratio = 10) reduces but does not eliminate the imbalance within training, and the internal early-stopping criterion of the MLP (validation accuracy) remains a coarse signal under imbalance.

**Future work.** Sequence-based models (LSTM, Transformer) applied to per-second approach trajectory data could exploit temporal structure that aggregate features discard. Oversampling techniques (SMOTE) or cost-sensitive boosting may further improve minority-class recovery. Airport-specific fine-tuning could address the performance variation observed across locations.

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
