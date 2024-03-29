#   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#   ┃                       1. Loading of required packages                        ┃
#   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
if (!require("fastDummies"))  install.packages("fastDummies")
if (!require("pheatmap"))     install.packages("pheatmap")
if (!require("ggplot2"))      install.packages("ggplot2")
if (!require("caret"))        install.packages("caret")
if (!require("nnet"))         install.packages("nnet")
if (!require("randomForest")) install.packages("randomForest")
if (!require("e1071"))        install.packages("e1071")
if (!require("kernlab"))      install.packages("kernlab")
library(fastDummies)
library(pheatmap)
library(ggplot2)
library(caret)
library(nnet)
library(randomForest)
library(e1071)
library(kernlab)
#    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
#   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#   ┃                             2. Data exploration                              ┃
#   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │                    2.1. Loading data and creating a copy                     │
#   └──────────────────────────────────────────────────────────────────────────────┘
filename <- "/Users/oliver/Library/Mobile Documents/com~apple~CloudDocs/FH Campus Wien/6. Semester/Programmkonzeption, Programmierung, Automatisierung, Bachelorarbeit/cirrhosis.csv"
if(file.exists(filename)) {
  data <- read.csv(filename)
} else {
  cat("File not found. Please choose the file.\n")
  data <- read.csv(file.choose())
}
data_expl <- data
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │                2.2. Showing the structure of the data set                    │
#   └──────────────────────────────────────────────────────────────────────────────┘
str(data_expl)
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │                 2.3. Showing the summary of the data set                     │
#   └──────────────────────────────────────────────────────────────────────────────┘
summary(data_expl)
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │       2.4. Determination of the number of missing values per variable        │
#   └──────────────────────────────────────────────────────────────────────────────┘
colSums(is.na(data_expl))
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │                   2.5. Removal of rows with missing values                   │
#   └──────────────────────────────────────────────────────────────────────────────┘
data_expl <- na.omit(data_expl)
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │             2.6. One-hot encoding for all categorical variables              │
#   └──────────────────────────────────────────────────────────────────────────────┘
data_expl <- fastDummies::dummy_cols(data_expl, select_columns = c("Status", "Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema", "Stage"), remove_first_dummy = FALSE)
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │                    2.7. Creation of a correlation matrix                     │
#   └──────────────────────────────────────────────────────────────────────────────┘
corMatrix <- cor(data_expl[sapply(data_expl, is.numeric)], use="pairwise.complete.obs")
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │ 2.8. Manual alphabetical sorting of the rows and columns of the correlation  │
#   │                                    matrix                                    │
#   └──────────────────────────────────────────────────────────────────────────────┘
orderedCorMatrix <- corMatrix[order(rownames(corMatrix)), order(colnames(corMatrix))]
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │      2.9. Creation of a heatmap / visualization of the correlation matrix    │
#   └──────────────────────────────────────────────────────────────────────────────┘
pheatmap(orderedCorMatrix, 
         display_numbers = TRUE,
         number_format = "%.2f",
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         color = colorRampPalette(c("blue", "white", "red"))(100),
         main = "Correlation Matrix")
#    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
#   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#   ┃                             3. Data preparation                              ┃
#   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │               3.1. Removal of variables that are not required                │
#   └──────────────────────────────────────────────────────────────────────────────┘
data <- data[ , !(names(data) %in% c("ID", "N_Days", "Drug", "Stage"))]
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │                   3.2. Removal of rows with missing values                   │
#   └──────────────────────────────────────────────────────────────────────────────┘
data <- na.omit(data)
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │3.3. Conversion of the categorical variables into factors, for later modeling │
#   └──────────────────────────────────────────────────────────────────────────────┘
data$Status <- as.factor(data$Status)
data$Sex <- as.factor(data$Sex)
data$Ascites <- as.factor(data$Ascites)
data$Hepatomegaly <- as.factor(data$Hepatomegaly)
data$Spiders <- as.factor(data$Spiders)
data$Edema <- as.factor(data$Edema)
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │            3.4. Conversion of the age (variable "Age") into years            │
#   └──────────────────────────────────────────────────────────────────────────────┘
data$Age <- data$Age / 365.25
#    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
#   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#   ┃    4. Splitting of the data set into training and test data (70/30 split)    ┃
#   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
set.seed(1234) #for reproducibility
index <- createDataPartition(data$Status, p=0.7, list=FALSE)
trainData <- data[index, ]
testData <- data[-index, ]
#    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
#   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#   ┃                 5. Model training (neural network with nnet)                 ┃
#   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
testData$Status <- factor(testData$Status, levels=levels(trainData$Status))
nnet_model <- nnet(Status ~ ., data=trainData, size=5, decay=0.1, maxit=200)
nnet_predictions <- predict(nnet_model, testData, type="class")
nnet_predictions <- factor(nnet_predictions, levels=levels(testData$Status))
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │                    5.1. Model evaluation (neural network)                    │
#   └──────────────────────────────────────────────────────────────────────────────┘
confusionMatrix(nnet_predictions, testData$Status)
cm <- confusionMatrix(nnet_predictions, testData$Status)
cm_table <- as.table(cm$table)
accuracy <- sum(diag(cm$table)) / sum(cm$table)
ggplot(data = as.data.frame(cm$table), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() + 
  geom_text(aes(label = sprintf("%0.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "aquamarine3") +
  theme_minimal() +
  labs(title = paste("Confusion Matrix - Neural Network\nAccuracy:", sprintf("%.2f%%", accuracy * 100)), 
       x = "Actual Outcome", y = "Predicted Outcome") +
  theme(axis.text.x = element_text(vjust = 1, hjust=1))
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │     5.2. Hyperparameter optimization and model training (neural network)     │
#   └──────────────────────────────────────────────────────────────────────────────┘
nnetGrid <- expand.grid(size = c(1, 5, 10), decay = c(0.1, 0.01, 0.001))
nnet_control <- trainControl(method = "cv", number = 10)
nnet_model_optimized <- caret::train(Status ~ ., data = trainData, method = "nnet",
                                     trControl = nnet_control, tuneGrid = nnetGrid, 
                                     linout = TRUE, trace = FALSE, maxit = 200)
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │              5.3. Model evaluation (neural network, optimized)               │
#   └──────────────────────────────────────────────────────────────────────────────┘
nnet_predictions_optimized <- predict(nnet_model_optimized, testData)
confusionMatrix(nnet_predictions_optimized, testData$Status)
cm <- confusionMatrix(nnet_predictions_optimized, testData$Status)
cm_table <- as.table(cm$table)
accuracy <- sum(diag(cm$table)) / sum(cm$table)
ggplot(data = as.data.frame(cm$table), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() + 
  geom_text(aes(label = sprintf("%0.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "aquamarine3") +
  theme_minimal() +
  labs(title = paste("Confusion Matrix - Neural Network (optimized)\nAccuracy:", sprintf("%.2f%%", accuracy * 100)), 
       x = "Actual Outcome", y = "Predicted Outcome") +
  theme(axis.text.x = element_text(vjust = 1, hjust=1))
#    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
#   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#   ┃             6. Model training (random forest with randomForest)              ┃
#   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
rf_model <- randomForest(Status ~ ., data=trainData, ntree=100)
rf_predictions <- predict(rf_model, testData)
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │                    6.1. Model evaluation (random forest)                     │
#   └──────────────────────────────────────────────────────────────────────────────┘
confusionMatrix(rf_predictions, testData$Status)
cm <- confusionMatrix(rf_predictions, testData$Status)
cm_table <- as.table(cm$table)
accuracy <- sum(diag(cm$table)) / sum(cm$table)
ggplot(data = as.data.frame(cm$table), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() + 
  geom_text(aes(label = sprintf("%0.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "aquamarine3") +
  theme_minimal() +
  labs(title = paste("Confusion Matrix - Random Forest\nAccuracy:", sprintf("%.2f%%", accuracy * 100)), 
       x = "Actual Outcome", y = "Predicted Outcome") +
  theme(axis.text.x = element_text(vjust = 1, hjust=1))
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │     6.2. Hyperparameter optimization and model training (random forest)      │
#   └──────────────────────────────────────────────────────────────────────────────┘
rfGrid <- expand.grid(mtry = c(2, 4, 6, 8))
rf_control <- trainControl(method="cv", number=10)
rf_model_optimized <- caret::train(Status ~ ., data = trainData, method = "rf",
                                   trControl = rf_control, tuneGrid = rfGrid)
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │               6.3. Model evaluation (random forest, optimized)               │
#   └──────────────────────────────────────────────────────────────────────────────┘
rf_predictions_optimized <- predict(rf_model_optimized, testData)
confusionMatrix(rf_predictions_optimized, testData$Status)
cm <- confusionMatrix(rf_predictions_optimized, testData$Status)
cm_table <- as.table(cm$table)
accuracy <- sum(diag(cm$table)) / sum(cm$table)
ggplot(data = as.data.frame(cm$table), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() + 
  geom_text(aes(label = sprintf("%0.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "aquamarine3") +
  theme_minimal() +
  labs(title = paste("Confusion Matrix - Random Forest (optimized)\nAccuracy:", sprintf("%.2f%%", accuracy * 100)), 
       x = "Actual Outcome", y = "Predicted Outcome") +
  theme(axis.text.x = element_text(vjust = 1, hjust=1))
#    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
#   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#   ┃          7. Model training (support vector machine, SVM, with e1071)         ┃
#   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
svm_model <- svm(Status ~ ., data=trainData, kernel="radial")
svm_predictions <- predict(svm_model, testData)
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │                         7.1. Model evaluation (SVM)                          │
#   └──────────────────────────────────────────────────────────────────────────────┘
confusionMatrix(svm_predictions, testData$Status)
cm <- confusionMatrix(svm_predictions, testData$Status)
cm_table <- as.table(cm$table)
accuracy <- sum(diag(cm$table)) / sum(cm$table)
ggplot(data = as.data.frame(cm$table), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() + 
  geom_text(aes(label = sprintf("%0.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "aquamarine3") +
  theme_minimal() +
  labs(title = paste("Confusion Matrix - SVM\nAccuracy:", sprintf("%.2f%%", accuracy * 100)), 
       x = "Actual Outcome", y = "Predicted Outcome") +
  theme(axis.text.x = element_text(vjust = 1, hjust=1))
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │          7.2. Hyperparameter optimization and model training (SVM)           │
#   └──────────────────────────────────────────────────────────────────────────────┘
svmGrid <- expand.grid(sigma = c(0.001, 0.01, 0.1), C = c(1, 10, 100))
svm_control <- trainControl(method="cv", number=10)
svm_model_optimized <- caret::train(Status ~ ., data=trainData, method="svmRadial",
                                    trControl=svm_control, tuneGrid=svmGrid)
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │                    7.3. Model evaluation (SVM, optimized)                    │
#   └──────────────────────────────────────────────────────────────────────────────┘
svm_predictions_optimized <- predict(svm_model_optimized, testData)
confusionMatrix(svm_predictions_optimized, testData$Status)
cm <- confusionMatrix(svm_predictions_optimized, testData$Status)
cm_table <- as.table(cm$table)
accuracy <- sum(diag(cm$table)) / sum(cm$table)
ggplot(data = as.data.frame(cm$table), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() + 
  geom_text(aes(label = sprintf("%0.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "aquamarine3") +
  theme_minimal() +
  labs(title = paste("Confusion Matrix - SVM (optimized)\nAccuracy:", sprintf("%.2f%%", accuracy * 100)), 
       x = "Actual Outcome", y = "Predicted Outcome") +
  theme(axis.text.x = element_text(vjust = 1, hjust=1))
#    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
