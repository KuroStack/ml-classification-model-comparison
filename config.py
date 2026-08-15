PAGE_TITLE = "ML Classification — BITS Pilani WILP"
PAGE_ICON = ":material/model_training:"

SIDEBAR_TITLE = ":material/model_training: ML Assignment 2"
SIDEBAR_CAPTION = "BITS Pilani WILP · AIML CZG565 · Machine Learning"
UPLOAD_CAPTION = "Upload the test CSV to evaluate all models on your data."
SIDEBAR_INFO = (
    "**Dataset:** UCI Cardiotocography\n\n"
    "**Instances:** 2 126\n\n"
    "**Features:** 21\n\n"
    "**Task:** Multiclass  \n(Normal / Suspect / Pathological)"
)

PAGE_HEADING = "ML Classification Models — Comparative Study"
PAGE_SUBTITLE = (
    "**BITS Pilani WILP · AIML CZG565 · Assignment 2** &nbsp;|&nbsp; "
    "Dataset: *UCI Cardiotocography* (dataset 193)"
)

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbor",
    "Naive Bayes (Gaussian)",
    "Random Forest (Ensemble)",
]

METRIC_COLS = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]

DATASET_TOTAL_INSTANCES = "2 126"
DATASET_FEATURE_COUNT = "21"

UPLOAD_PROMPT = "Upload `test_data.csv` in the sidebar to evaluate the selected model."

UPLOAD_NO_TARGET_WARNING = (
    "Uploaded CSV has no recognisable target column (`fetal_health` or `NSP`)."
)

CONFUSION_MATRIX_CAPTION = (
    "3-class confusion matrix: **Normal**, **Suspect**, **Pathological**."
)

DATASET_OVERVIEW_MD = """
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

**Our preprocessing :**
- Median impute is in the pipeline but UCI reports no missing values
- Z-score standardisation: $x_{std} = (x - \\mu) / \\sigma$ (fit on train only)
- Train / Test split: 80 % / 20 %, stratified
- Multiclass metrics: weighted Precision / Recall / F1; one-vs-rest AUC
"""

MODEL_NOTES = {
    "Logistic Regression": """
**Logistic Regression**

- Softmax
- Cost: cross-entropy
""",
    "Decision Tree": """
**Decision Tree**

- Gain: $\\text{Gain}(S, A) = H(S) - \\sum_{v} \\frac{|S_v|}{|S|} H(S_v)$
- Entropy: $H(S) = -\\sum_i p_i \\log_2 p_i$
""",
    "K-Nearest Neighbor": """
**k-Nearest Neighbor**

- Euclidean distance, $K = \\sqrt{{N}}$ → **K = {k_value}**
- Majority class among K neighbours
""",
    "Naive Bayes (Gaussian)": """
**Gaussian Naive Bayes**

- $P(X_1, \\ldots, X_n | Y) = \\prod_{j=1}^{n} P(X_j | Y)$
- $\\hat{Y} = \\arg\\max_{y_k} P(Y=y_k) \\cdot \\prod_i P(X_i | Y=y_k)$
""",
    "Random Forest (Ensemble)": """
**Random Forest**

- $B$ trees on bootstrap samples; $\\sqrt{n_{\\text{features}}}$ features per split
- Majority vote over trees
""",
}


def model_note(model_name: str, k_value: int | None = None) -> str:
    text = MODEL_NOTES.get(model_name, "")
    if model_name == "K-Nearest Neighbor":
        return text.format(k_value=k_value)
    return text

