## Dataset: QSAR fish toxicity

Source: UCI Machine Learning Repository <br>
Link: https://archive.ics.uci.edu/dataset/504/qsar+fish+toxicity <br>
UCI ID: 504 <br>
Domain: Physics and Chemistry / QSAR Toxicity Modeling <br>
Task: Tabular Regression <br>
Target Variable: `LC50`

The QSAR Fish Toxicity dataset contains molecular descriptor values for 908 chemicals, including CIC0, SM1_Dz(Z), GATS1i, NdsCH, NdssC, and MLOGP. The target used in this benchmark is `LC50`, which represents the concentration that causes death in 50% of test fish over a test duration of 96 hours. The target is reported as -log10(LC50) in mol/L. In this benchmark, we use the dataset as a regression task to compare traditional statistical models, kernel methods, Gaussian process regression, tree-based models, and deep learning models.