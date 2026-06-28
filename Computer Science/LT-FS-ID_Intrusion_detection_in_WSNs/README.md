## Dataset: LT-FS-ID: Intrusion detection in WSNs

**Source:** UCI Machine Learning Repository  
**Link:** https://archive.ics.uci.edu/dataset/715/lt+fs+id+intrusion+detection+in+wsns
**UCI ID:** 715
**Domain:** Wireless Sensor Networks / Cybersecurity / Intrusion Detection
**Task:** Tabular Regression  
**Target Variable:** `Number of Barriers`

The LT-FS-ID: Intrusion Detection in WSNs dataset contains wireless sensor network simulation variables related to intrusion detection and prevention. The dataset includes area size, sensing range, transmission range, and the number of deployed sensor nodes.

In this benchmark, we use the dataset as a regression task to predict `Number of Barriers`, which represents the number of barriers required for intrusion detection in a wireless sensor network. This dataset is included in the Wireless Sensor Networks / Cybersecurity / Intrusion Detection domain because it directly models a network-security planning outcome using sensor deployment and communication parameters.

Compared with medium-sized tabular regression datasets, LT-FS-ID is a small-scale dataset with 182 samples and 4 numerical features. This makes it useful for evaluating model behavior in low-sample and low-dimensional settings. In particular, this dataset helps examine whether deep learning models such as MLP and FT-Transformer provide clear advantages over classical methods such as Ridge Regression, SVR-RBF, Gaussian Process Regression, Random Forest, and Gradient Boosting when the amount of available data is limited.
