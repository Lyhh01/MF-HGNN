# MF-HGNN: A Multi-View Fusion Heterogeneous Graph Neural Network for Psychiatric Disorder Diagnosis.

This is a implementation of the FC-HGNN model as proposed in our paper. The code is available at https://github.com/Lyhh01/MF-HGNN.


## Introduction
Accurate and efficient diagnosis of psychiatric disorders remains a significant challenge in medicine. Graph Neural Networks (GNNs), known for their strength in modeling non-Euclidean data, find extensive application in the analysis of brain connectivity. While individual-level graph methods can detect abnormal brain regions linked to disorders, they often rely on static, single-view (i.e., non-temporal) data and fail to capture temporal dynamics. In contrast, population-level models can incorporate non-imaging data but often overlook site-specific heterogeneity, leading to poor accuracy and interpretability. In order to surmount the aforementioned issues, a two-stage end-to-end framework is proposed: Multi-View Fusion Heterogeneous Graph Neural Network (MF-HGNN). The first stage extracts individual brain features from spatiotemporal and topological views and identifies biomarkers using a random-walk aggregation pooling layer. The second stage integrates imaging and non-imaging data to build heterogeneous population graph (HPG) that reflects site-level diversity. A hierarchical aggregation strategy learns feature embeddings from both intra- and inter-site neighbors. We further introduce a dual-stage cascaded attention convolution module to extract shared and complementary features, followed by a global convolution for Autism Spectrum Disorder (ASD) classification. MF-HGNN achieves 95.07% accuracy and 97.36% AUC on ABIDE I, 99.16% accuracy and 99.86% AUC on Rest-MDD, outperforming existing methods. The identified biomarkers align with prior research, confirming the model’s reliability.

For more details about MF-HGNN, please refer to our paper.

## Instructions
The public datasets used in the paper can be obtained from their official sources. Running `main.py` trains the model and makes predictions. `main_transductive.py` adds a validation set at run time. When training and testing your own data, it is recommended to try adjusting the relevant hyperparameters.

## Requirements:
* torch
* torch_geometric
* scipy
* numpy
* os

# Repository Structure
```
├── ABIDE/                        # ABIDE 标准10折交叉验证（个体水平）
│   ├── run.log                   # 完整训练日志
│   ├── test.log                  # 完整测试日志
│   ├── ABIDE_Demo.ipynb          # 测试复现Demo
│   ├── dataload.py               # 数据集加载与划分（核心）
│   ├── main.py                   # 训练入口文件
│   └── model.py                  # 模型结构

├── ABIDE_LeaveGroupOut/          # ABIDE 留组外交叉验证
│   ├── run.log
│   ├── test.log
│   ├── ABIDE_LeaveGroupOut_Demo.ipynb
│   ├── dataload.py               # 数据集加载与留组划分（核心）
│   ├── main.py
│   └── model.py

├── ABIDE_SiteLeaveGroupOut/      # ABIDE 按站点留组外交叉验证
│   ├── run.log
│   ├── test.log
│   ├── ABIDE_SiteLeaveGroupOut_Demo.ipynb
│   ├── dataload.py               # 数据集加载与站点留组划分（核心）
│   ├── main.py
│   └── model.py

├── MDD/                          # MDD 标准10折交叉验证
│   ├── run.log
│   ├── test.log
│   ├── MDD_Demo.ipynb
│   ├── dataload.py               # 数据集加载与划分（核心）
│   ├── main.py
│   └── model.py

├── MDD_LeaveGroupOut/            # MDD 留组外交叉验证
│   ├── run.log
│   ├── test.log
│   ├── MDD_LeaveGroupOut_Demo.ipynb
│   ├── dataload.py               # 数据集加载与留组划分（核心）
│   ├── main.py
│   └── model.py

├── MDD_SiteLeaveGroupOut/        # MDD 按站点留组外交叉验证
│   ├── run.log
│   ├── test.log
│   ├── MDD_SiteLeaveGroupOut_Demo.ipynb
│   ├── dataload.py               # 数据集加载与站点留组划分（核心）
│   ├── main.py
│   └── model.py

├── model/                        # 公共模型核心代码
│   ├── brainmsgpassing.py
│   ├── base_model.py
│   └── metrics.py

└── README.md                    # 仓库说明文档
```

---

## 📦 Model Checkpoints (Baidu Netdisk)
All pre-trained model weights (`ckpt_demo/`) are **not uploaded to GitHub** due to large file sizes. They can be downloaded from:

**Link：MF-HGNN_Model_Checkpoints**
https://pan.baidu.com/s/1qII63kUEUc2tQDKOi-JwnQ
**Extract Code：t7t9**

After downloading, place the `ckpt_demo/` folder into the **corresponding experimental directory** to run the Jupyter demo.

---

## 📌 Key Files Introduction
- **`dataload.py`** (in each folder): Responsible for **dataset loading and data partitioning** (including 10-fold, leave-group-out, and site-level leave-group-out cross-validation).
- **`run.log` / `test.log`**: Complete training and testing records.
- **`*_Demo.ipynb`**: One-click reproduction demo for all experimental results.
