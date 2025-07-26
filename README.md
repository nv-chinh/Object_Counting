# Object Counting with CLIP-EBC

This project is based on the paper: **[CLIP-EBC: CLIP-Guided Efficient Box-Free Object Counting](https://arxiv.org/abs/2308.13741)** (ICCV 2023)

---

## 🔍 Overview

This project presents a cutting-edge object counting approach that integrates the rich semantic understanding of CLIP with the lightweight efficiency of the Enhanced Blockwise Classification (EBC) framework. By leveraging natural language prompts to guide the counting process, this method delivers accurate, adaptable, and prompt-driven object counting across diverse datasets and complex visual scenarios.

---

## 🧩 Features

- **CLIP-Guided Semantics**: Utilizes the power of CLIP to align visual features with language prompts for robust object understanding.

- **Enhanced Blockwise Classification (EBC)**: Improves traditional blockwise counting by introducing integer-valued bins and a joint classification-regression loss (DACE). This design eliminates boundary ambiguity and enables accurate count learning even in heavily crowded, overlapping, or small-object-dominated scenes.

- **Flexible Backbone Support**: Compatible with multiple architectures (ResNet, CLIP-ViT,...) - flexible for various deployment needs

- **Strong Quantitative Results**: Achieves competitive MAE, RMSE, and Accuracy metrics across multiple challenging benchmarks, especially in crowd counting scenarios.

- **Visual Interpretability**: Produces density heatmap to visualize model focus and counting regions.

---

## 📊 Experiments on Datasets

We evaluate our CLIP-EBC approach on several challenging object counting benchmarks to demonstrate its robustness and generalization:

### 🧍‍♂️ Crowd Counting – [ShanghaiTech Part A & B](https://github.com/desenzhou/ShanghaiTechDataset)

- **Scenario**: High-density crowd scenes with severe occlusion and scale variation.
- **Results**: CLIP-EBC shows consistent improvements in MAE and RMSE compared to prior box-free approaches, with especially notable accuracy in extremely dense environments.
- **Visualization**: The generated heatmaps effectively highlight dense regions, providing interpretable evidence of the model’s counting rationale.

### 🐮 Animal Counting – [Cow Counting Dataset](https://github.com/TrentBrown/CowCounting)

- **Scenario**: Real-world farm surveillance with sparse and medium-density cow clusters under variable lighting and backgrounds.
- **Results**: CLIP-EBC achieves high counting accuracy and demonstrates strong generalization without needing dataset-specific tuning, thanks to CLIP's semantic alignment.
- **Prompt Used**: `"a photo of a cow"`

### 🌳 Tree Counting – [DeepTreeCount Dataset](https://github.com/AdeelMufti/DeepTreeCount)

- **Scenario**: Aerial images of forest regions with partially occluded or overlapping trees.
- **Results**: The model accurately counts both isolated and dense tree regions. The EBC structure handles varied spatial distributions with ease.
- **Prompt Used**: `"a top view of a tree"`

### 🦐 Shrimp Counting – [Shrimp Dataset (internal)]()

- **Scenario**: Microscopic or underwater imagery with small-scale, clustered objects.
- **Results**: CLIP-EBC effectively suppresses noise and background artifacts, aided by EBC’s blockwise density learning and CLIP’s fine-grained semantics.
- **Prompt Used**: `"a shrimp underwater"`
