# Evaluation of Datasets for Outlier Detection

by Felippe P. Ferreira, and Robson L. F. Cordeiro

## Abstract

> Progress in outlier detection algorithms relies on robust validation, which is supported by high-quality datasets. However, the common practice of adapting datasets for this purpose can introduce flaws, falsely simulating instances as outliers. Despite the large number of available datasets from different subjects, are they suitable for outlier detection tasks? Several datasets used are adaptations of classification problems, containing classes that are not imbalanced, differing from the characteristics of a dataset with outliers. This paper aims to evaluate datasets widely used in detection algorithm comparisons, seeking to identify: (i) which datasets perform worst when subjected to outlier detection algorithms; (ii) analyze in detail the worst-performing datasets by comparing the labels assigned to the instances; and (iii) verify whether specific dataset adaptation strategies can impact algorithm performance. Through experiments with 52 datasets cited in various scientific studies, along with 22 outlier detection algorithms, we found that some datasets present ground truth inconsistent with the results obtained and the characteristics of an outlier instance. The instances in these problematic datasets exhibit characteristics not expected of outlier instances. Furthermore, the use of techniques such as downsampling in conjunction with class clustering produces datasets with better outlier detection results. To aid the experiments, a methodology was developed to ensure an objective evaluation of each dataset.

## Overview

This project provides a comprehensive analysis of the properties of popular benchmark datasets used in outlier detection research. While most research focuses on comparing *algorithms*, this work focuses on evaluating the *datasets*.

The study evaluates 52 commonly used benchmark datasets from outlier detection publications, submitted to 22 outlier detection algorithms, identifying problems in the labels assigned to some instances.

To conduct the experiments, a methodology was created to identify datasets that presented problems in the ground-truth attributed to the instances. In addition to the results provided by the selected algorithms, cluster analyses of the instances and analyses of the dataset creation process were performed.

Complementarily, an analysis was conducted on methods used to transform datasets originally for classification into outlier detection datasets.

## Key Contributions

This research offers three main contributions to the field of outlier detection:

### 1\. Dataset Ground Truth Evaluation

As our main contribution, the results of the evaluation of 52 datasets that were used in 22 detection algorithms show that some of the datasets, widely used in research in the area, have characteristics that are not consistent with the anomaly detection approach, in addition to producing results, together with the algorithms, inferior to other datasets. Through comparisons, it is possible to identify datasets that present problems, preventing algorithms from performing detections correctly.

### 2\. Process Methodology

To evaluate this large set of datasets and algorithms, a methodology was developed to simplify the evaluation process for each dataset, allowing for comparison of results. The methodology features a sequential execution flow that enables us to verify both objective aspects related to the dataset’s characteristics and subjective elements associated with information from the data creation process.

### 3\. Analysis of Dataset Conversion Strategies

Given the large number of datasets originating from classification tasks with no pronounced imbalance of the classes, this paper carried out a comparative analysis of techniques for converting classification data into outlier-detection data, aiming to identify if there is any technique that produces datasets of better quality than others. Our results indicate that downsampling with class grouping usually outperforms other options.


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

## Sections

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

  * Outlier Detection
  * Data Quality
  * Dataset Evaluation

## Authors

* **Felippe Pires Ferreira**
    * Institute of Mathematical and Computer Sciences, University of São Paulo (USP), São Carlos, SP, Brazil.
    * *Email: felippe_pires@usp.br*
* **Robson L. F. Cordeiro**
    * Institute of Mathematical and Computer Sciences, University of São Paulo (USP), São Carlos, SP, Brazil.
