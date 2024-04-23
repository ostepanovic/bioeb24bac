# [FH Campus Wien - BIOEB24] </br> Coding part of the bachelor thesis of Oliver Stepanovic
## Task
The task is to train different machine learning algorithms based on a dataset (<a href="https://www.kaggle.com/datasets/joebeachcapital/cirrhosis-patient-survival-prediction/data">link to the dataset on kaggle.com</a>), which contains clinical and laboratory data of cirrhosis patients, to predict the outcome of patients with cirrhosis (multi-class classification problem).
Different machine learning models will be trained and optimized to improve the prediction accuracy. The performance of the models is evaluated by creating confusion matrices and calculating the accuracies.
## Description of the uploaded R code
<a href="https://github.com/ostepanovic/bioeb24bac/blob/b54a84161d8a33ea65840378e55a7e24cfceb152/v1.0_Bac-Arbeit.R">v1.0_Bac-Arbeit.R</a> contains my own original code to accomplish the task. </br>

<a href="https://github.com/ostepanovic/bioeb24bac/blob/b54a84161d8a33ea65840378e55a7e24cfceb152/v1.1_Bac-Arbeit_mit_Datenaufbereitung.R">v1.1_Bac-Arbeit_mit_Datenaufbereitung.R</a> contains my modified code with the data preparation suggested on Kaggle:
+ Drop all the rows where miss value (NA) were present in the Drug column
+ Impute missing values with mean results
+ One-hot encoding for all category attributes

<a href="https://github.com/ostepanovic/bioeb24bac/blob/b54a84161d8a33ea65840378e55a7e24cfceb152/v2.0_Bac-Thesis.R">v2.0_Bac-Thesis.R</a> contains further modified code with changes suggested by my supervisor, the most notable being:
+ Addition of a correlation matrix to the section "Data exploration" in order to visualize the impact of the variables on patient outcome
+ Removal of variables "ID", "N_Days", "Drug" and "Stage" from the data used for training of the machine learning algorithms
+ Removal of rows with missing values (instead of imputation) prior to algorithm training
+ Conversion of the variable "Age" into years

<a href=https://github.com/ostepanovic/bioeb24bac/blob/4c742c9ac1bc00e6a137da374ce6768c7e8d2ddd/v3.0_Bac-Thesis.R>v3.0_Bac-Thesis.R</a> contains the following changes:
+ Addition of initialization of random seed at the start for consistent results
+ Removal of rows with patient status "CL" (= censored because of liver transplantation)
+ Undoing the removal of the "Stage" variable (compared to v2.0)
+ Improvement of hyperparameter optimization
+ Addition of visualization of feature importance for each of the machine learning algorithms used

<a href=https://github.com/ostepanovic/bioeb24bac/blob/5596b4a8e9d317a8004590cdb1a6f92c9d9f2cf3/v3.1_Bac-Thesis.R>v3.1_Bac-Thesis.R</a> has been changed compared to v3.0 so that the variable "Drug" is predicted instead of the variable "Status".

<a href=https://github.com/ostepanovic/bioeb24bac/blob/5596b4a8e9d317a8004590cdb1a6f92c9d9f2cf3/v3.2_Bac-Thesis.R>v3.2_Bac-Thesis.R</a> has been changed compared to v3.0 so that the variable "Stage" is predicted instead of the variable "Status".
