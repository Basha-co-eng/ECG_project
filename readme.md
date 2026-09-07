# Uncertainty-Aware Explainable Federated Learning for Multi-Label ECG Classification

## Overview
This project presents a privacy-preserving framework for multi-label ECG classification using Federated Learning, Explainable Artificial Intelligence (XAI), and Uncertainty Quantification.

The framework investigates whether predictive uncertainty can serve as an indicator of explanation consistency across multiple healthcare institutions.

---

## Key Features
- **Multi-client Federated Learning**: Decentralized training simulating distinct hospital nodes without sharing patient raw ECG recordings.
- **PhysioNet Challenge 2021 Dataset**: Large-scale multi-hospital standard 12-lead ECG dataset.
- **1D ResNet-34 Architecture**: Specialized 1D deep residual network designed for multi-lead electrophysiological time-series.
- **Federated Optimization Algorithms**: Comparison across **FedAvg**, **FedProx** (proximal regularization for non-IID data), and **FedAdam** (adaptive server optimization).
- **Grad-CAM Explainability**: Lead-wise and temporal salience activation maps highlighting diagnostic cardiac wave segments.
- **SHAP Feature Attribution**: Tree-based Shapley additive explanations on clinical fiducial features.
- **Monte Carlo Dropout Uncertainty Estimation**: Epistemic predictive uncertainty quantification across client test partitions.
- **Cross-Client Explanation Consistency Analysis**: Pairwise cosine similarity quantification of explanations across clinical institutions.
- **Correlation Analysis**: Rigorous Pearson & Spearman correlation validating predictive uncertainty as an indicator of explanation reliability.

---

## Dataset
**PhysioNet Challenge 2021 (CinC 2021)**

### Federated Clients (Hospitals / Sources):
1. **Chapman-Shaoxing** (Shaoxing People's Hospital, China)
2. **CPSC-2018** (China Physiological Signal Challenge 2018)
3. **Georgia** (Emory University, Atlanta, USA)
4. **Ningbo** (Ningbo First Hospital, China)
5. **PTB-XL** (Physikalisch-Technische Bundesanstalt, Germany)

### Final Harmonized Taxonomy (12 Classes):
| Acronym | Condition Name | SNOMED CT Code |
|:---|:---|:---|
| **AF** | Atrial Fibrillation | 164889003 |
| **IAVB** | First-degree Atrioventricular Block | 270492004 |
| **LAD** | Left Axis Deviation | 39732003 |
| **LBBB** | Left Bundle Branch Block | 164909002 / 733534002 |
| **NSIVCB** | Nonspecific Intraventricular Conduction Disorder | 698252002 |
| **NSR** | Normal Sinus Rhythm | 426783006 |
| **PAC** | Premature Atrial Contraction | 284470004 |
| **QAb** | Abnormal Q Wave | 164917005 |
| **RBBB** | Right Bundle Branch Block | 59118001 |
| **SB** | Sinus Bradycardia | 426177001 |
| **STach** | Sinus Tachycardia | 427084000 |
| **TAb** | T Wave Abnormal | 164934002 |

---

## Methodology
```text
Raw 12-Lead ECG
       ↓
Preprocessing (Bandpass Filter, Resampling to 500Hz, Normalization)
       ↓
Label Harmonization (SNOMED-CT Mapping to 12 Target Classes)
       ↓
Federated Learning (5 Non-IID Hospital Clients)
       ↓
1D ResNet-34 Feature Extractor & Multi-Label Classifier
       ↓
Optimization: FedAvg / FedProx / FedAdam
       ↓
XAI Layer: Grad-CAM (Temporal/Lead Saliency) + SHAP (Fiducial Attribution)
       ↓
Monte Carlo Dropout (30 Forward Passes for Epistemic Uncertainty)
       ↓
Consistency Analysis (Inter-Client Cosine Explanation Consistency)
       ↓
Correlation Analysis (Uncertainty vs. Inconsistency Assessment)
```

---

## Repository Structure
```text
ECG-FL-XAI/
│
├── README.md                           # Project documentation & benchmark overview
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
│
├── data/
│   ├── class_mapping.csv               # 12-class SNOMED CT clinical taxonomy mapping
│   ├── client_statistics.csv           # Hospital dataset distribution & sample counts
│   └── sample_data/                    # Sample ECG recordings & schema documentation
│
├── preprocessing/
│   ├── dataset_downloading.py          # Header ingestion & PhysioNet metadata inventory
│   ├── label_harmonization.py          # SNOMED multi-label encoding & taxonomy mapping
│   └── feature_extraction.py           # Waveform ETL, filtering, & fiducial feature calculation
│
├── models/
│   ├── resnet34_1d.py                  # 1D ResNet-34 architecture for multi-lead ECG
│   ├── fedavg.py                       # Federated Averaging (FedAvg) aggregation logic
│   ├── fedprox.py                      # FedProx algorithm with proximal regularization
│   └── fedadam.py                      # FedAdam / FedOpt adaptive server optimizer
│
├── training/
│   ├── train_fedavg.py                 # Multi-round federated training pipeline for FedAvg
│   ├── train_fedprox.py                # Multi-round federated training pipeline for FedProx
│   └── train_fedadam.py                # Multi-round federated training pipeline for FedAdam
│
├── explainability/
│   ├── gradcam.py                      # 1D Grad-CAM implementation for ECG time-series
│   ├── shap_analysis.py                # SHAP TreeExplainer & feature attribution pipeline
│
├── uncertainty/
│   ├── mc_dropout.py                   # Monte Carlo Dropout epistemic uncertainty quantification
│
├── analysis/
│   ├── consistency_analysis.py         # Inter-client explanation cosine consistency evaluator
│   ├── correlation_analysis.py         # Statistical correlation between uncertainty & XAI consistency
│
├── results/
│   ├── classification_results.csv      # Macro/Micro F1 scores across clients & algorithms
│   ├── consistency_scores.csv          # Explanation consistency scores per model & class
│   ├── uncertainty_scores.csv          # Predictive entropy & MC variance metrics
│   ├── correlation_results.csv         # Pearson & Spearman correlation statistics
│   ├── central_history.csv             # Centralized baseline convergence log
│   ├── fedavg_micro_macro_full_history.csv  # FedAvg round-by-round convergence log
│   ├── fedprox_round_history.csv       # FedProx round-by-round convergence log
│   └── fedopt_round_history.csv        # FedAdam round-by-round convergence log
│
├── figures/
│   ├── architecture.png                # System & network architectural block diagram
│   ├── gradcam_examples/               # Sample 1D Grad-CAM salience visual explanations
│   ├── shap_examples/                  # SHAP summary & beeswarm plots (AF, NSR)
│   └── correlation_plots/              # Convergence curves, scatter plots, & heatmaps
│
└── notebooks/
    ├── preprocessing.ipynb             # Interactive data cleaning, filtering, & inspection
    ├── training.ipynb                  # Interactive federated learning training workflow
    ├── gradcam.ipynb                   # Interactive Grad-CAM heatmap visualization
    ├── shap.ipynb                      # Interactive SHAP fiducial importance computation
    └── analysis.ipynb                  # Interactive uncertainty-consistency correlation analysis
```

---

## Experimental Pipeline
1. **ECG Preprocessing**: Bandpass filtering (0.5Hz - 45Hz), resampling to 500Hz, amplitude normalization, and length standardization (5000 samples / 10s per lead).
2. **Client Partitioning**: Multi-center hospital allocation preserving realistic non-IID real-world clinical distributions.
3. **Federated Training**: 30 communication rounds evaluating FedAvg, FedProx (mu = 0.001), and FedAdam (beta1=0.9, beta2=0.99, tau=1e-3).
4. **Explainability Generation**: 1D Grad-CAM feature attribution over convolutional activation maps and SHAP on cardiological fiducial metrics.
5. **Uncertainty Estimation**: 30 Monte Carlo Dropout stochastic passes per sample measuring predictive entropy, mutual information, and variance.
6. **Consistency & Correlation Analysis**: Quantitative evaluation of explanation agreement across hospital nodes and statistical correlation against uncertainty.

---

## Experimental Results

### 1. Classification Performance (Test Micro/Macro F1)
| Method | Chapman-Shaoxing (F1) | CPSC-2018 (F1) | Georgia (F1) | Ningbo (F1) | PTB-XL (F1) | Mean Macro F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Centralized Baseline** | 0.814 | 0.781 | 0.745 | 0.808 | 0.822 | **0.794** |
| **FedAvg** | 0.699 | 0.310 | 0.592 | 0.756 | 0.718 | **0.615** |
| **FedProx (mu=0.001)** | 0.748 | 0.364 | 0.635 | 0.789 | 0.752 | **0.658** |
| **FedAdam** | **0.767** | **0.612** | **0.546** | **0.808** | **0.774** | **0.701** |

### 2. Explanation Consistency & Uncertainty Correlation
- **Inter-Client Consistency**: FedAdam and FedProx demonstrated significantly higher explanation agreement across non-IID client partitions compared to vanilla FedAvg.
- **Uncertainty Correlation**: Strong, statistically significant positive correlation (r = 0.684, p < 0.01) observed between predictive uncertainty (entropy/spread) and cross-client explanation inconsistency.
- **Clinical Insight**: When the federated model exhibits elevated predictive uncertainty on ambiguous ECG morphologies, its explanatory saliency maps diverge across institutions, providing a robust built-in safety indicator for clinical decision support.

---

## Getting Started

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended for federated training)

### Installation
```bash
git clone https://github.com/Md-SAMI-7/ECG-FL-XAI.git
cd ECG-FL-XAI
pip install -r requirements.txt
```

### Running the Pipeline
```bash
# 1. Preprocess & download dataset metadata
python preprocessing/dataset_downloading.py
python preprocessing/label_harmonization.py

# 2. Train Federated Models
python training/train_fedavg.py
python training/train_fedprox.py
python training/train_fedadam.py

# 3. Generate Explainability Maps
python explainability/gradcam.py
python explainability/shap_analysis.py

# 4. Quantify Uncertainty & Compute Correlation
python uncertainty/mc_dropout.py
python analysis/consistency_analysis.py
python analysis/correlation_analysis.py
```

---

## Authors
**Team Members:**
- **Syed Mohammad Samiul Ahmed**
- **Shaik Mahammad Galeeb**
- **Shaik Khasim Basha**
- **Shaik Rehaman**

**Project Guide:**
- **Dr. Lalitha Kumari P**
- School of Computer Science and Engineering (SCOPE)
- **VIT-AP University**

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.