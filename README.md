# Evaluation of Datasets for Outlier Detection

by Felippe P. Ferreira, and Robson L. F. Cordeiro

## Abstract

> Progress in outlier detection algorithms relies on robust validation, which is supported by high-quality datasets. However, the common practice of adapting datasets for this purpose can introduce flaws, falsely simulating instances as outliers. Despite the large number of available datasets from different subjects, are they suitable for outlier detection tasks? Several datasets used are adaptations of classification problems, containing classes that are not imbalanced, differing from the characteristics of a dataset with outliers. This paper aims to evaluate datasets widely used in detection algorithm comparisons, seeking to identify: (i) which datasets perform worst when subjected to outlier detection algorithms; (ii) analyze in detail the worst-performing datasets by comparing the labels assigned to the instances; and (iii) verify whether specific dataset adaptation strategies can impact algorithm performance. Through experiments with 47 datasets cited in various scientific studies, along with 22 outlier detection algorithms, we found that some datasets present ground truth inconsistent with the results obtained and the characteristics of an outlier instance. The instances in these problematic datasets exhibit characteristics not expected of outlier instances. Furthermore, the use of techniques such as downsampling in conjunction with class clustering produces datasets with better outlier detection results. To aid the experiments, a methodology was developed to ensure an objective evaluation of each dataset.

## Overview

This project provides a comprehensive analysis of the properties of popular benchmark datasets used in outlier detection research. While most research focuses on comparing *algorithms*, this work focuses on evaluating the *datasets*.

The study evaluates 47 commonly used benchmark datasets from outlier detection publications, submitted to 27 outlier detection algorithms, identifying problems in the labels assigned to some instances.

To conduct the experiments, a methodology was created to identify datasets that presented problems in the ground-truth attributed to the instances. In addition to the results provided by the selected algorithms, cluster analyses of the instances and analyses of the dataset creation process were performed.

Complementarily, an analysis was conducted on methods used to transform datasets originally for classification into outlier detection datasets.

## Directory Tree

A summary of the file structure can be found in the following directory tree.

```bash
EvaluateDatasets
├── files                  \\ Main container for all project assets.
│   ├── code               \\ Contains all source code and analysis scripts.
│   │   ├── profiling      \\ Scripts to extract meta-features from each dataset.
│   │   │   ├── Run_Profiler.ipynb   \\ Notebook for calculating statistical metrics (dim, sparsity, etc.).
│   │   │   └── Calculate_Metrics.py \\ Script for calculating data complexity metrics.
│   │   │
│   │   ├── benchmarking   \\ Scripts to run probe algorithms on the datasets.
│   │   │   ├── Run_LOF.py
│   │   │   ├── Run_IForest.py
│   │   │   └── ...
│   │   │
│   │   └── analysis       \\ Jupyter notebooks for analyzing the profiling results.
│   │       ├── Cluster_Datasets.ipynb \\ Notebook for clustering datasets by similarity.
│   │       ├── Plot_Difficulty.ipynb  \\ Notebook for generating difficulty vs. outlier type plots.
│   │       └── Redundancy_Matrix.ipynb \\ Notebook for creating the rank correlation matrix.
│   │
│   ├── database           \\ Stores all datasets used in the experiments.
│   │   ├── raw            \\ Original datasets downloaded from their sources.
│   │   │   ├── uci
│   │   │   ├── kaggle
│   │   │   └── ...
│   │   └── processed      \\ Pre-processed and standardized datasets (.npz format).
│   │
│   └── results            \\ Stores all output files from the experiments.
│       ├── profiles       \\ Metrics and meta-features extracted from each dataset.
│       │   └── dataset_metrics.csv \\ The profile of all 85 datasets.
│       │
│       ├── benchmarks     \\ Raw results of the probe algorithms on each dataset.
│       │
│       └── analysis       \\ Plots and tables generated for the paper.
│           ├── CLUSTERING   \\ Dendrogram images of dataset clustering.
│           ├── DIFFICULTY_PLOTS \\ Difficulty scatter plots.
│           └── tables       \\ Ranking tables and the proposed "core set".
│
└── README.md              \\ Project overview, setup instructions, and documentation.
```

## Key Contributions

This research offers three main contributions to the field of outlier detection:

### 1\. Dataset Ground Truth Evaluation

As our main contribution, the results of the evaluation of 47 datasets that were used in 22 detection algorithms show that some of the datasets, widely used in research in the area, have characteristics that are not consistent with the anomaly detection approach, in addition to producing results, together with the algorithms, inferior to other datasets. Through comparisons, it is possible to identify datasets that present problems, preventing algorithms from performing detections correctly.

### 2\. Process Methodology

To evaluate this large set of datasets and algorithms, a methodology was developed to simplify the evaluation process for each dataset, allowing for comparison of results. The methodology features a sequential execution flow that enables us to verify both objective aspects related to the dataset’s characteristics and subjective elements associated with information from the data creation process.

### 3\. Analysis of Dataset Conversion Strategies

Given the large number of datasets originating from classification tasks with no pronounced imbalance of the classes, this paper carried out a comparative analysis of techniques for converting classification data into outlier-detection data, aiming to identify if there is any technique that produces datasets of better quality than others. Our results indicate that downsampling with class grouping usually outperforms other options.

## Execution Instructions

This repository contains the scripts to analyze the datasets. The original (raw) datasets are not included, but the download scripts can be found in `files/database/raw/`. As a first step, it is necessary to run the profiling and benchmarking scripts:

PrepareDataset.ipynb
Evaluate.py
Metrics.ipynb
PairPlot_metrics.ipynb

  * **Run\_Profiler.ipynb**
  * **Calculate\_Metrics.py**

The first script normalizes the datasets (storing them in `files/database/processed/`) and the second generates the `dataset_metrics.csv` file (in `files/results/profiles/`), which is the basis for all other analyses.

For each processed dataset, the scripts in `files/code/benchmarking/` must be run to generate the raw results of the probe algorithms. The results should be stored in `files/results/benchmarks/`.

To group the results and generate the final analyses, the scripts in `files/code/analysis` should be executed in the following order:

  * **Cluster\_Datasets.ipynb**
  * **Redundancy\_Matrix.ipynb**
  * **Plot\_Difficulty.ipynb**

The **Cluster\_Datasets.ipynb** file uses the `dataset_metrics.csv` and benchmark results to generate the dataset clusters and identify the "Core Set". The **Redundancy\_Matrix.ipynb** generates the rank correlation diagrams. The last file, **Plot\_Difficulty.ipynb**, trains the regression model and plots the datasets on a "Difficulty" vs. "Outlier Type" chart.

## Keywords

  * Outlier Detection
  * Data Quality
  * Dataset Evaluation

## Authors

* **Felippe Pires Ferreira**
    * Institute of Mathematical and Computer Sciences, University of São Paulo (USP), São Carlos, SP, Brazil.
    * *Email: felippe_pires@usp.br*
* **Robson L. F. Cordeiro**
    * School of Computer Science, Carnegie Mellon University (CMU), Pittsburgh, PA, USA.
    * *Email: robsonc@andrew.cmu.edu*