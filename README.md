
# Applications of Deep Learning in Statistical Modeling

This repository contains the code and experimental pipeline for our project on **applications of deep learning in statistical modeling**, with a focus on **tabular regression tasks**.

The goal of this project is not to propose a new model. Instead, we aim to conduct a **survey and empirical comparison** of representative statistical learning, machine learning, and deep learning methods under a unified experimental setting.

## Project Motivation

Traditional statistical models such as Linear Regression and Ridge Regression are widely used because they are simple, interpretable, and efficient. However, many real-world regression problems contain nonlinear relationships and feature interactions that may not be well captured by linear models.

Deep learning models, such as Multi-Layer Perceptrons and Transformer-based architectures, provide more flexible function approximation. However, recent studies show that deep learning does not always outperform traditional machine learning methods on tabular data. Tree-based models such as Random Forest and Gradient Boosting often remain highly competitive.

Therefore, this project investigates the following question:

> Do deep learning models provide clear advantages over classical statistical and machine learning methods for tabular regression?

## Compared Models

The current benchmark includes the following regression models:

| Category                      | Model                          |
| ----------------------------- | ------------------------------ |
| Linear Statistical Model      | Linear Regression              |
| Regularized Statistical Model | Ridge Regression               |
| Kernel Method                 | SVR with RBF Kernel            |
| Bayesian Method               | Gaussian Process Regression    |
| Tree Ensemble                 | Random Forest                  |
| Gradient Boosting             | HistGradientBoosting / XGBoost |
| Deep Learning Baseline        | Multi-Layer Perceptron         |
| Tabular Deep Learning         | FT-Transformer                 |

## Current Test Dataset

The current pipeline is tested on the **UCI Energy Efficiency Dataset**.

This dataset is a tabular regression dataset used to predict building energy performance. The current target is:

* `Y1`: Heating Load

The pipeline can also be modified to predict:

* `Y2`: Cooling Load

## Evaluation Metrics

All models are evaluated on the same fixed train-test split using the following metrics:

* MAE
* RMSE
* R² Score
* Training Time

The fixed test split allows a fair comparison between different model families.

## Pipeline Overview

The pipeline performs the following steps:

1. Load the dataset
2. Preprocess numerical and categorical features
3. Split the data into training and test sets
4. Train all selected regression models
5. Evaluate each model on the same fixed test set
6. Save numerical results as CSV files
7. Generate comparison figures

## Output Files

After running the pipeline, the output folder will contain:

```text
benchmark_outputs_energy_efficiency/
├── results/
│   ├── model_results.csv
│   ├── test_predictions.csv
│   └── test_indices.csv
└── figures/
    ├── 01_r2_sorted.png
    ├── 02_mae_rmse_grouped.png
    ├── 03_training_time_log.png
    ├── 04_predicted_vs_true_grid.png
    ├── 05_sorted_test_predictions.png
    ├── 06_residual_boxplot.png
    └── 07_error_vs_true_best_model.png
```

## How to Run

### Option 1: Use Conda Environment

Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml -n tabular_benchmark
```

Activate the environment:

```bash
conda activate tabular_benchmark
```

Run the pipeline:

```bash
python pipeline.py
```

### Option 2: Use requirements.txt

Install required packages:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python pipeline.py
```

## Notes

If `ucimlrepo` is missing, install it with:

```bash
pip install ucimlrepo
```

If `xgboost` is missing, install it with:

```bash
pip install xgboost
```

If XGBoost is not installed, the pipeline can still run without it, depending on the code settings.

## Current Findings

On the Energy Efficiency dataset, nonlinear models such as HistGradientBoosting, XGBoost, Random Forest, and Gaussian Process Regression achieve very strong performance. Linear Regression and Ridge Regression perform worse, which suggests that this dataset contains nonlinear relationships that are difficult for simple linear models to capture.

The FT-Transformer model also achieves competitive performance, but it requires more training time and does not clearly outperform tree-based models on this dataset. This supports the broader observation that deep learning models are not always superior on medium-sized tabular regression datasets.

## Future Work

Future extensions of this project include:

* Adding more datasets from different domains
* Expanding the benchmark to around 10 domains with multiple datasets per domain
* Comparing model performance across small, medium, and larger datasets
* Adding more tabular deep learning models such as TabNet
* Performing more systematic hyperparameter tuning
* Writing a survey section discussing statistical models, kernel methods, Bayesian learning, tree-based models, and deep learning for regression

## Project Positioning

This project is designed as a **survey plus empirical study**. The main contribution is a unified comparison of different regression model families on tabular datasets, rather than a new model architecture.
