#   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#   ┃                          1. benötigte Pakete laden                           ┃
#   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
library(ggplot2)
library(caret)
library(nnet)
library(randomForest)
library(e1071)
library(kernlab)
#    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
#   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#   ┃                             2. Datenexploration                              ┃
#   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │                               2.1. Datensatz laden                           │
#   └──────────────────────────────────────────────────────────────────────────────┘
daten <- read.csv("/Users/oliver/Library/Mobile Documents/com~apple~CloudDocs/FH Campus Wien/6. Semester/Programmkonzeption, Programmierung, Automatisierung, Bachelorarbeit/cirrhosis.csv")
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │                    2.2. Struktur des Datensatzes anzeigen                    │
#   └──────────────────────────────────────────────────────────────────────────────┘
str(daten)
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │                2.3. Zusammenfassung des Datensatzes anzeigen                 │
#   └──────────────────────────────────────────────────────────────────────────────┘
summary(daten)
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │            2.4. Anzahl der fehlenden Werte pro Variable ermitteln            │
#   └──────────────────────────────────────────────────────────────────────────────┘
colSums(is.na(daten))
#    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
#   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#   ┃                        3. Behandlung fehlender Werte                         ┃
#   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │  3.1. Entfernen aller Zeilen, in denen in der Spalte "Drug" fehlende Werte   │
#   │                                vorhanden sind                                │
#   └──────────────────────────────────────────────────────────────────────────────┘
daten <- daten[!is.na(daten$Drug), ]
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │3.2. Ersetzen fehlender Werte für numerische Variablen durch das arithmetische│
#   │                                    Mittel                                    │
#   └──────────────────────────────────────────────────────────────────────────────┘
numerische_vars <- c("N_Days", "Age", "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin")
for(col in numerische_vars) {
  daten[[col]][is.na(daten[[col]])] <- mean(daten[[col]], na.rm = TRUE)
}
#    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
#   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#   ┃             4. One-Hot-Encoding für alle kategorialen Variablen              ┃
#   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
daten <- fastDummies::dummy_cols(daten, select_columns = c("Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema", "Stage"), remove_first_dummy = TRUE)
#    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
#   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#   ┃     5. Konvertierung der kategorialen Variablen in Faktoren, für spätere     ┃
#   ┃         Modellierung (fast alle redundant, aufgrund One-Hot-Encoding)        ┃
#   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
daten$Status <- as.factor(daten$Status)
#daten$Drug <- as.factor(daten$Drug)
#daten$Sex <- as.factor(daten$Sex)
#daten$Ascites <- as.factor(daten$Ascites)
#daten$Hepatomegaly <- as.factor(daten$Hepatomegaly)
#daten$Spiders <- as.factor(daten$Spiders)
#daten$Edema <- as.factor(daten$Edema)
#daten$Stage <- as.factor(daten$Stage)
#    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
#   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#   ┃  6. Standardisierung bzw. Normalisierung der numerischen Variablen (vorerst  ┃
#   ┃                                  optional)                                   ┃
#   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
numerische_vars <- c("N_Days", "Age", "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin")

for(col in numerische_vars) {
  daten[[col]] <- scale(daten[[col]])
}
#    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
#   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#   ┃    7. Splitten des Datensatzes in Trainings- und Testdaten (70/30-Split)     ┃
#   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
set.seed(1234) #für Reproduzierbarkeit
index <- createDataPartition(daten$Status, p=0.7, list=FALSE)
trainData <- daten[index, ]
testData <- daten[-index, ]
#    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
#   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#   ┃                 8. Modelltraining (neuronales Netz mit nnet)                 ┃
#   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
testData$Status <- factor(testData$Status, levels=levels(trainData$Status))
nnet_model <- nnet(Status ~ ., data=trainData, size=5, decay=0.1, maxit=200)
nnet_predictions <- predict(nnet_model, testData, type="class")
nnet_predictions <- factor(nnet_predictions, levels=levels(testData$Status))
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │                    8.1. Modellbewertung (neuronales Netz)                    │
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
  labs(title = paste("Confusion Matrix - neuronales Netz\nAccuracy:", sprintf("%.2f%%", accuracy * 100)), 
       x = "Tatsächlicher Wert", y = "Vorhergesagter Wert") +
  theme(axis.text.x = element_text(vjust = 1, hjust=1))
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │     8.2. Hyperparameter-Optimierung und Modelltraining (neuronales Netz)     │
#   └──────────────────────────────────────────────────────────────────────────────┘
nnetGrid <- expand.grid(size = c(1, 5, 10), decay = c(0.1, 0.01, 0.001))
nnet_control <- trainControl(method = "cv", number = 10)
nnet_model_optimized <- caret::train(Status ~ ., data = trainData, method = "nnet",
                                     trControl = nnet_control, tuneGrid = nnetGrid, 
                                     linout = TRUE, trace = FALSE, maxit = 200)
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │              8.3. Modellbewertung (neuronales Netz, optimiert)               │
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
  labs(title = paste("Confusion Matrix - neuronales Netz (optimiert)\nAccuracy:", sprintf("%.2f%%", accuracy * 100)), 
       x = "Tatsächlicher Wert", y = "Vorhergesagter Wert") +
  theme(axis.text.x = element_text(vjust = 1, hjust=1))
#    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
#   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#   ┃              9. Modelltraining (Random Forest mit randomForest)              ┃
#   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
rf_model <- randomForest(Status ~ ., data=trainData, ntree=100)
rf_predictions <- predict(rf_model, testData)
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │                     9.1. Modellbewertung (Random Forest)                     │
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
       x = "Tatsächlicher Wert", y = "Vorhergesagter Wert") +
  theme(axis.text.x = element_text(vjust = 1, hjust=1))
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │      9.2. Hyperparameter-Optimierung und Modelltraining (Random Forest)      │
#   └──────────────────────────────────────────────────────────────────────────────┘
rfGrid <- expand.grid(mtry = c(2, 4, 6, 8))
rf_control <- trainControl(method="cv", number=10)
rf_model_optimized <- caret::train(Status ~ ., data = trainData, method = "rf",
                                   trControl = rf_control, tuneGrid = rfGrid)
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │               9.3. Modellbewertung (Random Forest, optimiert)                │
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
  labs(title = paste("Confusion Matrix - Random Forest (optimiert)\nAccuracy:", sprintf("%.2f%%", accuracy * 100)), 
       x = "Tatsächlicher Wert", y = "Vorhergesagter Wert") +
  theme(axis.text.x = element_text(vjust = 1, hjust=1))
#    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
#   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#   ┃          10. Modelltraining (Support Vector Machine, SVM, mit e1071)         ┃
#   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
svm_model <- svm(Status ~ ., data=trainData, kernel="radial")
svm_predictions <- predict(svm_model, testData)
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │                         10.1. Modellbewertung (SVM)                          │
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
       x = "Tatsächlicher Wert", y = "Vorhergesagter Wert") +
  theme(axis.text.x = element_text(vjust = 1, hjust=1))
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │          10.2. Hyperparameter-Optimierung und Modelltraining (SVM)           │
#   └──────────────────────────────────────────────────────────────────────────────┘
svmGrid <- expand.grid(sigma = c(0.001, 0.01, 0.1), C = c(1, 10, 100))
svm_control <- trainControl(method="cv", number=10)
svm_model_optimized <- caret::train(Status ~ ., data=trainData, method="svmRadial",
                                    trControl=svm_control, tuneGrid=svmGrid)
#   ┌──────────────────────────────────────────────────────────────────────────────┐
#   │                   10.3. Modellbewertung (SVM, optimiert)                     │
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
  labs(title = paste("Confusion Matrix - SVM (optimiert)\nAccuracy:", sprintf("%.2f%%", accuracy * 100)), 
       x = "Tatsächlicher Wert", y = "Vorhergesagter Wert") +
  theme(axis.text.x = element_text(vjust = 1, hjust=1))
#    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
