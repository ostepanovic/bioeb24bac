# [FH Campus Wien - BIOEB24] </br> Coding part of the bachelor thesis of Oliver Stepanovic
## Task
The task is to train different machine learning algorithms based on a dataset (<a href="https://www.kaggle.com/datasets/joebeachcapital/cirrhosis-patient-survival-prediction/data">link to the dataset on kaggle.com</a>), which contains clinical and laboratory data of cirrhosis patients, to predict the outcome of patients with cirrhosis (multi-class classification problem).
Different machine learning models will be trained and optimized to improve the prediction accuracy. The performance of the models is evaluated by creating confusion matrices and calculating the accuracies.
## Description of the uploaded R code
<a href="https://github.com/ostepanovic/bioeb24bac/blob/4d145120660229e4ffccd547602bec1ec424ab52/Bac-Arbeit_v1.0.R">Bac-Arbeit_v1.0.R</a> contains my own original code to accomplish the task. </br>
</br>
<a href="https://github.com/ostepanovic/bioeb24bac/blob/4d145120660229e4ffccd547602bec1ec424ab52/Bac-Arbeit_mit_Datenaufbereitung_v1.1.R">Bac-Arbeit_mit_Datenaufbereitung_v1.1.R</a> contains my modified code with the data preparation suggested on Kaggle:
+ Drop all the rows where miss value (NA) were present in the Drug column
+ Impute missing values with mean results
+ One-hot encoding for all category attributes
