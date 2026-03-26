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
