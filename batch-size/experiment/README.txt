This folder has the datasets and the code for experimenting with different batch sizes. From this experiment, we wanted to measure the impact of different batch sizes on the workload memory estimates, using LearnedWMP approach. 

In this experiment, I used TPC-DS dataset, where queries were assigned into 110 templates. These 110 templates were learned using a separate notebook with K-Means clustering. 

I used 13 values for the batch size: 1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50

For each value of the batch size, I have two datasets: a training dataset and a test dataset. They are in the data folder. 
Along with these two datasets, I created a notebook for training a model using the training dataset and evaluating the trained model using the test dataset. 
For the model implementation, I used XGB model. As for evaluating the model's performance, I used Mean Absolute Percent Error (MAPE) metric. 

I manaully copied the MAPE values from each experiment notebook, into the excel file: Book1.xlsx

To-Dos:
2. I manually added lines to the Book1.xlsx file. It would be nice to automatically updating this file with MAPE value from each notebook. 