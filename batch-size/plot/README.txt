This folder has the code for plotting results from the batch size experiment. The experiment results come from the other folder in the same parent directory of this folder: batch-size/experiment

This folder has 3 files, besides this README file:
batchsize_4.xlsx: this file has two columns
(1) batch sizes: [2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
(2) MAPE score for each batch size

batch-size-experiment-results-MAPE-plot.ipynb
- this notebook loads data from batchsize_4.xlsx and generates a plot that shows the changes in MAPE as we varied the batch size. The code also saves the plot into "Batch Sizes.png" file. 



