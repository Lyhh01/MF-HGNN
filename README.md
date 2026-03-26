好的，我帮你把**Dataset Partitioning Descriptions**这一段格式统一成和前面完全一致的风格，整体 README 保持整洁规范，你指定保留的部分完全不动，只统一排版：

---

# MF-HGNN: A Multi-View Fusion Heterogeneous Graph Neural Network for Psychiatric Disorder Diagnosis.

This is an implementation of the FC-HGNN model as proposed in our paper. The code is available at https://github.com/Lyhh01/MF-HGNN.

## Introduction
Accurate and efficient diagnosis of psychiatric disorders remains a significant challenge in medicine. Graph Neural Networks (GNNs), known for their strength in modeling non-Euclidean data, find extensive application in the analysis of brain connectivity. While individual-level graph methods can detect abnormal brain regions linked to disorders, they often rely on static, single-view (i.e., non-temporal) data and fail to capture temporal dynamics. In contrast, population-level models can incorporate non-imaging data but often overlook site-specific heterogeneity, leading to poor accuracy and interpretability. In order to surmount the aforementioned issues, a two-stage end-to-end framework is proposed: Multi-View Fusion Heterogeneous Graph Neural Network (MF-HGNN). The first stage extracts individual brain features from spatiotemporal and topological views and identifies biomarkers using a random-walk aggregation pooling layer. The second stage integrates imaging and non-imaging data to build heterogeneous population graph (HPG) that reflects site-level diversity. A hierarchical aggregation strategy learns feature embeddings from both intra- and inter-site neighbors. We further introduce a dual-stage cascaded attention convolution module to extract shared and complementary features, followed by a global convolution for Autism Spectrum Disorder (ASD) classification. MF-HGNN achieves 95.07% accuracy and 97.36% AUC on ABIDE I, 99.16% accuracy and 99.86% AUC on Rest-MDD, outperforming existing methods. The identified biomarkers align with prior research, confirming the model’s reliability.

For more details about MF-HGNN, please refer to our paper.

## Instructions
The public datasets used in the paper can be obtained from their official sources. Running `main.py` trains the model and makes predictions. `main_transductive.py` adds a validation set at run time. When training and testing your own data, it is recommended to try adjusting the relevant hyperparameters.

## Requirements
* torch
* torch_geometric
* scipy
* numpy
* os

## Repository Structure
```
├── ABIDE/                        # ABIDE dataset: standard 10-fold cross-validation
│   ├── run.log                   # Full training log
│   ├── test.log                  # Full testing log
│   ├── ABIDE_Demo.ipynb          # Jupyter demo for result reproduction
│   ├── dataload.py               # Dataset loading and data splitting
│   ├── main.py                   # Training script
│   └── model.py                  # Model architecture

├── ABIDE_LeaveGroupOut/          # ABIDE dataset: leave-group-out cross-validation stratified by site and diagnostic status
│   ├── run.log
│   ├── test.log
│   ├── ABIDE_LeaveGroupOut_Demo.ipynb
│   ├── dataload.py               
│   ├── main.py
│   └── model.py

├── ABIDE_SiteLeaveGroupOut/      # ABIDE dataset: site-specific leave-group-out cross-validation
│   ├── run.log
│   ├── test.log
│   ├── ABIDE_SiteLeaveGroupOut_Demo.ipynb
│   ├── dataload.py               
│   ├── main.py
│   └── model.py

├── MDD/                          # MDD dataset: standard 10-fold cross-validation (individual-level)
│   ├── run.log
│   ├── test.log
│   ├── MDD_Demo.ipynb
│   ├── dataload.py               
│   ├── main.py
│   └── model.py

├── MDD_LeaveGroupOut/            # MDD dataset: leave-group-out cross-validation stratified by site and diagnostic status
│   ├── run.log
│   ├── test.log
│   ├── MDD_LeaveGroupOut_Demo.ipynb
│   ├── dataload.py               
│   ├── main.py
│   └── model.py

├── MDD_SiteLeaveGroupOut/        # MDD dataset: site-specific leave-group-out cross-validation
│   ├── run.log
│   ├── test.log
│   ├── MDD_SiteLeaveGroupOut_Demo.ipynb
│   ├── dataload.py               
│   ├── main.py
│   └── model.py

├── MF-HGNN/        # Best Model

├── model/                        # model components
│   ├── brainmsgpassing.py
│   ├── base_model.py
│   └── metrics.py

└── README.md                     # Repository documentation
```

## 📦 Model Checkpoints (Baidu Netdisk)
All pre-trained model weights (`ckpt_demo/`) are **not uploaded to GitHub** due to large file sizes. They can be downloaded from:

**Link：MF-HGNN_Model_Checkpoints**
https://pan.baidu.com/s/1qII63kUEUc2tQDKOi-JwnQ
**Extract Code：t7t9**

After downloading, place the `ckpt_demo/` folder into the **corresponding experimental directory** to run the Jupyter demo.

## 📌 Key Files Introduction
- **`dataload.py`** (in each folder): Responsible for **dataset loading and data partitioning** (including 10-fold, stratified leave-group-out by site and diagnosis, and site-level leave-group-out cross-validation).
- **`run.log` / `test.log`**: Complete training and testing records.
- **`*_Demo.ipynb`**: One-click reproduction demo for all experimental results.

## Dataset Partitioning Descriptions
The following cross-validation strategies are applied to both ABIDE and MDD datasets.

* **Standard 10-Fold Cross-Validation (Individual-Level)**  
  Standard 10-fold stratified cross-validation at the individual level. Uses StratifiedKFold with stratification by diagnostic label (ASD vs. HC for ABIDE; MDD vs. HC for MDD).

* **Leave-Group-Out Cross-Validation (Stratified by Site and Diagnosis)**  
  10-fold leave-group-out cross-validation stratified by the joint distribution of diagnosis and site (DX_GROUP + SITE_ID). Ensures balanced group composition to reduce data leakage and enable rigorous generalization evaluation.

* **Leave-Group-Out (Site) Cross-Validation**  
  Strict leave-group-out cross-validation at the site level. This is a more rigorous validation strategy for evaluating model generalization across different imaging centers.
  - ABIDE: conducted on the three largest sites (NYU, UM, UCLA).
  - MDD: each fold tests on one entire site and trains on all others.
