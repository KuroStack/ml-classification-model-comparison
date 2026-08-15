# Machine Learning Model Comparison [🔗](https://ml-classification-model-comparison.streamlit.app/)

## a. Problem statement

Build an end-to-end multiclass classification pipeline on a public dataset, train five classification models on the same data, and evaluate each model with Accuracy, AUC, Precision, Recall, F1 and MCC.

## b. Dataset description

**Public repository:** [UCI Cardiotocography](https://archive.ics.uci.edu/dataset/193/cardiotocography)

Imported with `ucimlrepo.fetch_ucirepo(id=193)`.

The dataset consists of measurements of fetal heart rate (FHR) and uterine contraction (UC) features on cardiotocograms classified by expert obstetricians.

| | |
|---|---|
| Characteristics | Multivariate |
| Subject area | Health and Medicine |
| Associated tasks | Classification |
| Feature type | Real |
| Instances | 2126 |
| Features | 21 |
| Missing values | No |

2126 fetal cardiotocograms (CTGs) were automatically processed and the respective diagnostic features measured. The CTGs were also classified by three expert obstetricians and a consensus classification label assigned to each of them. Classification was both with respect to a morphologic pattern (A, B, C, …) and to a fetal state (N, S, P). Therefore the dataset can be used either for 10-class or 3-class experiments.

This assignment uses the **3-class** target **NSP** (N = Normal, S = Suspect, P = Pathological).

UCI variables table (Description and Units are empty on the source page):

| Variable | Role | Type | Missing |
|---|---|---|---|
| LB | Feature | Integer | no |
| AC | Feature | Continuous | no |
| FM | Feature | Continuous | no |
| UC | Feature | Continuous | no |
| DL | Feature | Continuous | no |
| DS | Feature | Continuous | no |
| DP | Feature | Continuous | no |
| ASTV | Feature | Integer | no |
| MSTV | Feature | Continuous | no |
| ALTV | Feature | Integer | no |
| MLTV | Feature | Continuous | no |
| Width | Feature | Integer | no |
| Min | Feature | Integer | no |
| Max | Feature | Integer | no |
| Nmax | Feature | Integer | no |
| Nzeros | Feature | Integer | no |
| Mode | Feature | Integer | no |
| Mean | Feature | Integer | no |
| Median | Feature | Integer | no |
| Variance | Feature | Integer | no |
| Tendency | Feature | Integer | no |
| CLASS | Target | Integer | no |
| NSP | Target | Integer | no |

## c. Github Repository Link

https://github.com/KuroStack/ml-classification-model-comparison

## d. Models used

All five models are trained on the same UCI Cardiotocography (id=193) data. Evaluation is the stratified 20% held-out split (426 rows), also saved as `test_data.csv`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8967 | 0.9702 | 0.8962 | 0.8967 | 0.8964 | 0.7184 |
| Decision Tree | 0.9343 | 0.9027 | 0.9328 | 0.9343 | 0.9334 | 0.8181 |
| kNN | 0.8873 | 0.9655 | 0.8814 | 0.8873 | 0.8814 | 0.6714 |
| Naive Bayes | 0.8286 | 0.9316 | 0.8754 | 0.8286 | 0.8423 | 0.6124 |
| Random Forest (Ensemble) | 0.9484 | 0.9861 | 0.9470 | 0.9484 | 0.9474 | 0.8567 |

### Performance Observation

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy 0.8967, AUC 0.9702. Suspect is the weak class (F1 ≈ 0.63). |
| Decision Tree | Accuracy 0.9343, MCC 0.8181. AUC 0.9027 is lower than logistic and kNN. |
| kNN | K = √N → 41. Accuracy 0.8873. Pathological recall 0.66; Suspect F1 ≈ 0.59. |
| Naive Bayes | Weakest overall (Accuracy 0.8286, MCC 0.6124). Over-predicts Suspect. |
| Random Forest (Ensemble) | Best on Accuracy 0.9484, AUC 0.9861, F1 0.9474, MCC 0.8567. |

Overall winner for the sample dataset is **Random Forest**
