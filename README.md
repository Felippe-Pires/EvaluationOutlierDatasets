# Evaluation of Datasets for Outlier Detection

by Felippe P. Ferreira, and Robson L. F. Cordeiro

## Abstract

> How to collect and preprocess a dataset to support the development and validation of outlier de-
tection techniques? How suitable are the datasets often used in the literature for this purpose?
Progress in outlier detection relies on robust validation supported by datasets with high-quality
ground-truth labels. However, many labeled datasets available in public repositories contain more
than two classes and do not present natural class imbalance, which deviates from the characteris-
tics of a dataset with outliers. A common practice is to adapt these datasets for outlier detection, by
generating simulated outliers through downsampling, grouping or exclusion of classes and the use
of other techniques that often introduce errors. This paper aims to answer the two questions above,
by: (i) evaluating datasets widely used to validate outlier detection algorithms, and; (ii) comparing
the most common adaptation techniques to learn best practices for data collection and prepro-
cessing in outlier detection scenarios. Our main contributions are: C1 – Evaluation of Datasets:
we assess 50 datasets that are largely employed and famous in the literature through an extensive
experimental evaluation supported by 22 of the best-known detectors of outliers. Our results re-
veal a list of datasets presenting questionable ground truth with potential negative effects in any
algorithm validated on these data. C2 – Best Practices: we evaluate the impact of common data
adaptation techniques on the ground-truth label quality, and learn that the use of downsampling
combined with grouping of classes often outperforms other options. C3 – Methodology: we also
introduce a novel methodology to evaluate any new dataset for outlier detection. It features a se-
quential execution flow that enables us to verify both objective aspects related to the dataset’s char-
acteristics and subjective elements associated with the data creation process. Our work supports
future progress in outlier detection by providing valuable guidelines for the selection, generation
and preprocessing of data used to develop new algorithms or validate the existing ones.

## Overview

This work proposes a methodology for assessing the reliability of ground-truth labels in outlier-detection benchmarks. The experimental study evaluates 50 datasets using 22 outlier-detection algorithms representing 10 detection principles. It also examines strategies for converting classification datasets into anomaly-detection benchmarks, using downsampling, class grouping, class exclusion, and combinations of grouping or exclusion with downsampling. The results show that creating class imbalance does not necessarily produce genuine anomalies. By combining Binary Evaluation with visual, neighborhood-density, and dataset-provenance analyses, the methodology identified 12 datasets whose labels received limited empirical support and are therefore not recommended for algorithm validation without further investigation. Four of these datasets—wpbc, vertebral, wilt, and amazon—were examined in detail to investigate how their origin, preprocessing, and conversion procedures may have affected the reliability of their labels.

## Key Contributions

This research offers three main contributions to the field of outlier detection:

### 1\. Evaluation of Datasets

We assess the quality of 47 datasets that are largely employed and famous in the literature through an extensive experimental evaluation supported by 22 of the best-known outlier detectors. Our results reveal a list of datasets having questionable ground truth with potential negative effects in any algorithm validated on these data. Hence, we argue that the datasets in our list should no longer be used by the outlier detection community.

### 2\. Best Practices

We evaluate the impact of common data adaptation techniques on the ground-truth label quality of the datasets, such as grouping and excluding classes and downsampling their instances. Our experimental results indicate that the use of downsampling combined with grouping of classes often outperforms other options.

### 3\. Methodology

We also introduce a novel methodology to evaluate any new dataset for outlier detection. It features a sequential execution flow that enables us to verify both objective aspects related to the dataset’s characteristics and subjective elements associated with the data creation process. Our methodology simplifies and standardizes the evaluation of a dataset and
also allows comparing the results obtained from distinct datasets.

## Directory Tree

A summary of the file structure can be found in the following directory tree.

```bash
EvaluateDatasets
├── files                       \\ Main container for all project assets.
│   ├── code                    \\ Contains all source code and analysis scripts.
│   │   ├── base_experiments    \\ Script for evaluating datasets.
│   │   │   ├── Evaluate.py     \\ Execution of detection algorithms on the datasets.
│   │   │   ├── Metrics.ipynb   \\ Calculation of metrics based on algorithm results.
│   │   │   ├── PairPlots_metrics.ipynb \\ Generating pair plots with the algorithm results.
│   │   │   ├── PrepareDataset.ipynb    \\ Script to prepare datasets before algorithm processing.
│   │   │   └── Visualization.ipynb     \\ Script to produce comparative tables of instances and projections of the worst-performing datasets.
│   │   │
│   │   └── conversion_methods              \\ Script for experiments using methods to convert classification datasets into outlier detection datasets.
│   │       ├── Convert_Methods.ipynb       \\ Apply the conversion methods to the datasets.
│   │       ├── Critical_Diagram.ipynb      \\ Compare the methods using critical difference diagrams.
│   │       ├── EvaluateConvert.py          \\ Apply the detection algorithms to the converted datasets.
│   │       └── Metrics.ipynb               \\ Calculation of metrics based on algorithm results.
│   │    
│   │
│   ├── datasets                    \\ Stores all datasets used in the experiments.
│   │   ├── conversion_methods      \\ Datasets used in experiments with conversion methods
│   │   │   ├── binary              \\ BIN and BINDOWN
│   │   │   └── non_binary          \\ EXC, EXCDOWN, GRO and GRODOWN
│   │   |
│   │   └── base_experiments              \\ Datasets used in the experiments to evaluate the quality of the outlier detection datasets.
│   │       ├── ADBench
│   │       ├── literature
│   │       ├── odds
│   │       ├── processed           \\ Directory for storing datasets after preprocessing.
│   │       ├── real
│   │       └── semantic
│   │
│   └── results                     \\ Stores all output files from the experiments. The directory is populated as the scripts are executed.
│       ├── conversion_methods      \\ Results of processing datasets submitted to conversion methods.
│       │   |── binary 
│       │   └── non_binary 
│       │
│       ├── instances_detected     \\ Result of the identified instances performed by each algorithm.
|       |
│       ├── pair_plot              \\ Pair plots from the dataset evaluation 
│       │
│       └── visualization_output   \\ Projection of datasets with the worst results.

│
└── README.md              \\ Project overview, setup instructions, and documentation.
```

## Repository Organization

The project sections were divided according to the experiments performed. To facilitate locating the information, a brief description of the information stored in the directories is provided below:

### > code

This section is for storing the code used in the project. To organize the source code, two main subdirectories were created:  **base_experiments** and **conversion_methods**.

- **base_experiments**
	- This directory contains: a set of codes that perform the experiments described in this research, as described in __Section 4.2 - Essential Flow__ of the related research article, and detailed in __Section 5.2 - Essential Flow: results__ of the same article.

- **conversion_methods**
	- This directory stores the source code for experiments on methods for converting classification datasets into outlier detection datasets. The methods used are described in __Section 1 - Introduction__. The conversion methods evaluated are:

    -> __DOWN__ (Class Downsample): One or more minority classes are selected as potential anomalies, and their instances are reduced to simulate outlying cases.

    -> __EXC__ (Class Exclusion): Some classes are removed from the original dataset to obtain a binary representation;

    -> __GRO__ (Class Grouping): some classes are merged to transform a multiclass dataset into a binary one;    
    

### > datasets

This section is dedicated to storing the datasets used in this research. The list of datasets for the base experiments is presented in __Section 5.1. - Dataset Selection__, and the datasets used in the analisys of conversion methods are listed in __Section 5.4. Complementary Analysis – Conversion of Classification Data__ of the article.

### > result

Output directory for scripts in the **base_experiments** and **conversion_methods** directories.

## Execution Instructions

This repository contains the scripts to analyze the datasets. Versions of the original datasets are included in directory `files/datasets/`. As a first step, it is necessary to run the scripts (**base_experiments**) in this order:

  * **PrepareDataset.ipynb**
  * **Evaluate.py**
  * **Metrics.ipynb**
  * **PairPlot_metrics.ipynb**
  * **Visualization.ipynb**

The first script normalizes the datasets (storing them in `files/datasets/evaluation/processed/`) and the second script (`Evaluate.py`) executes all algorithms over the datasets. The third script `Metrics.ipynb` create the metrics used in this research. The last script create plots to visualize the datas.

There is a second set of scripts responsible for analyzing the conversion methods:

  * **Convert_Methods.ipynb**
  * **EvaluateConvert.py**
  * **Metrics.ipynb**
  * **Critical_Diagram.ipynb**

Initially, it is necessary to apply the normalization processes and the application of the conversion methods (**Convert_Methods.ipynb**). Subsequently, the `EvaluateConvert.py` script executes the algorithms on the subset of datasets in the **files\datasets\conversion_methods** directory. The `Metrics.ipynb` script measures the metric values of the results of each algorithm, and the `Critical_Diagram.ipynb` script creates the representation of the critical difference diagrams (CDD).


## Keywords

  * Data Quality
  * Outlier Detection
  * Dataset Evaluation
  * Best Practices for Data Collection and Preprocessing

## Authors

* **Felippe Pires Ferreira**
    * Institute of Mathematical and Computer Sciences, University of São Paulo (USP), São Carlos, SP, Brazil.
    * *Email: felippe_pires@usp.br*
* **Robson L. F. Cordeiro**
    * Institute of Mathematical and Computer Sciences, University of São Paulo (USP), São Carlos, SP, Brazil.
