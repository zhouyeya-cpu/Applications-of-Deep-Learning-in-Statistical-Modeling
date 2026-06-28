## Dataset: Appliances Energy Prediction

**Source:** UCI Machine Learning Repository  
**Link:** https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
**UCI ID:** 374
**Domain:** Energy / ComputComputer Science / Building Energy Management
**Task:** Tabular Regression  
**Target Variable:** `Appliances`

The Appliances Energy Prediction dataset contains energy-use and environmental sensor measurements collected from a low-energy residential building. The dataset includes indoor temperature and humidity measurements from different rooms, outdoor weather variables, lighting energy use, and additional random variables.

In this benchmark, we use the dataset as a regression task to predict `Appliances`, which represents the energy consumption of household appliances. This dataset is included in the Energy / Smart Home / Building Energy Management domain because it directly models residential appliance energy use based on indoor environmental conditions, outdoor weather information, and household energy-related variables.

Compared with smaller tabular regression datasets such as LT-FS-ID, the Appliances Energy Prediction dataset is larger, with 19,735 observations and real-valued sensor features. Although the original data are time-stamped, this benchmark treats each timestamp as an independent tabular observation. This dataset helps examine whether deep learning models such as MLP and FT-Transformer become more competitive on larger tabular regression datasets compared with classical methods such as Ridge Regression, SVR-RBF, Gaussian Process Regression, Random Forest, and Gradient Boosting.
