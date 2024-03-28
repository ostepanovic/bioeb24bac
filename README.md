# [FH Campus Wien - BIOEB24] </br> Code-Teil der Bachelor-Arbeit von Oliver Stepanovic
## Aufgabenstellung
Bei der gestellten Aufgabe geht es darum, verschiedene Machine-Learning-Algorithmen auf Basis eines Datensatzes (<a href="https://www.kaggle.com/datasets/joebeachcapital/cirrhosis-patient-survival-prediction/data">Link zum Datensatz auf kaggle.com</a>), welcher klinische und labortechnische Daten von Zirrhose-Patienten beinhaltet, zu trainieren um das Outcome bei Patienten mit Zirrhose vorherzusagen (Mehrklassen-Klassifizierungsproblem). </br>
Verschiedene Machine-Learning-Modelle werden trainiert und optimiert, um die Vorhersagegenauigkeit zu verbessern. Die Leistung der Modelle wird durch die Erstellung von Konfusionsmatrizes und der Berechnung der Genauigkeiten bewertet.
## Beschreibung des hochgeladenen R-Codes
<a href="https://github.com/ostepanovic/bioeb24bac/blob/4d145120660229e4ffccd547602bec1ec424ab52/Bac-Arbeit_v1.0.R">Bac-Arbeit_v1.0.R</a> beinhaltet meinen eigenen ursprünglichen Code zur Bewältigung der Aufgabenstellung. </br>
</br>
<a href="https://github.com/ostepanovic/bioeb24bac/blob/4d145120660229e4ffccd547602bec1ec424ab52/Bac-Arbeit_mit_Datenaufbereitung_v1.1.R">Bac-Arbeit_mit_Datenaufbereitung_v1.1.R</a> beinhaltet meinen abgeänderten Code mit der auf Kaggle vorgeschlagenen Datenvorbereitung:
+ Drop all the rows where miss value (NA) were present in the Drug column
+ Impute missing values with mean results
+ One-hot encoding for all category attributes
