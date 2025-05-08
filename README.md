# Early Detection of Post-Hepatectomy Liver Failure Using Deep Learning

Welcome to version 1.0 of the project! This is the initial release, and more updates are coming soon.

This repository is based on the study:
**"Learning-based early detection of post-hepatectomy liver failure using temporal perioperative data: a nationwide multicenter retrospective study in China"**
Published in eClinicalMedicine, 2025.

[Access the article here 📄](https://www.sciencedirect.com/science/article/pii/S258953702500152X)

## 💡 Background

Post-hepatectomy liver failure (PHLF) is a leading cause of postoperative mortality after liver surgery. However, early detection remains clinically challenging due to the lack of real-time diagnostic tools. This study develops a deep learning approach to detect PHLF within the first 24 hours after surgery, using rich perioperative data and leveraging recent advances in AI.


## 📊 Study Design

A retrospective multicenter study was conducted on **1,832 patients** from six hospitals across China. We incorporated **temporal perioperative data**, including preoperative, intraoperative, and early postoperative (24h) variables. An additional **Western cohort of 242 patients** was obtained from the **MIMIC-IV** database for generalizability analysis.

### 🖼️ Study Design Overview

![Figure 1: Study Design](figure/Fig_overview_structure.pdf)

## 🧠 Model Architecture

Our model integrates:
- **Bio-Clinical BERT**: Encodes clinical variable names, time, and values as modular token sequences.
- **Context-aware Transformer**: Captures intra-variable and inter-variable temporal dynamics.
- **Two classifiers**:
  - A primary classifier for binary PHLF detection,
  - A hierarchical gated classifier for PHLF severity grading (Grade A/B/C).

## 📈 Key Results

- ✅ **Internal Validation (China)**: AUC = **0.952**
- 🌍 **External Validation (China)**: AUC = **0.884**

The architecture is designed to flexibly handle missing data and feature inconsistency across hospitals.

Our model outperforms 11 state-of-the-art ML/DL methods (e.g., XGBoost, TabNet, TransTab), and improves clinician performance in early PHLF detection (AUC 0.778 vs. 0.637).

## 🔍 Model Interpretability

- **SHAP** analysis identifies top predictors:  
  Postoperative PT-INR, number of liver segments resected, HBV infection, cirrhosis, and major hepatectomy.
- **t-SNE** visualizations show effective temporal feature encoding.
- The model is interpretable, with prediction rationale aligning with known clinical risk factors.

- ## 🏥 Clinical Impact

- Enables early clinical intervention within **24h post-op**
- Performs robustly with **incomplete or phase-specific inputs**
- Effectively detects **Grade B/C (clinically relevant)** PHLF cases
- Assists clinicians and enhances risk stratification

This approach shows great promise in improving postoperative management and rescue of critical patients.

## 📬 Data Access

- **Chinese multicenter data**: Available upon request from the corresponding authors due to privacy regulations.
- **MIMIC-IV dataset**: Publicly available at [https://physionet.org](https://physionet.org).

---

## 📚 Citation

> Wang K, Yang Q, Li K, *et al.*  
> **Learning-based early detection of post-hepatectomy liver failure using temporal perioperative data: a nationwide multicenter retrospective study in China.**  
> *eClinicalMedicine*. 2025;83:103220.  
> DOI: [10.1016/j.eclinm.2025.103220](https://doi.org/10.1016/j.eclinm.2025.103220)


