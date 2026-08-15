import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from config import (
    CONFUSION_MATRIX_CAPTION,
    MODEL_NAMES,
    PAGE_HEADING,
    PAGE_ICON,
    PAGE_TITLE,
    SIDEBAR_CAPTION,
    SIDEBAR_TITLE,
    UPLOAD_CAPTION,
    UPLOAD_NO_TARGET_WARNING,
)
from data import FetalHealthDataLoader
from evaluation import EvaluationMetrics
from model import (
    DecisionTreeModel,
    KNNModel,
    LogisticRegressionModel,
    NaiveBayesModel,
    RandomForestModel,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Importing UCI Cardiotocography (id=193)…")
def _load_data():
    loader = FetalHealthDataLoader(test_size=0.20, random_state=42)
    X_train, X_test, y_train, y_test, _feature_names = loader.load()
    return loader, X_train, X_test, y_train, y_test


@st.cache_resource(show_spinner="Training all 5 models…")
def _train_all_models(_X_train, _y_train, n_classes):
    instances = {
        "Logistic Regression": LogisticRegressionModel(max_iter=1000),
        "Decision Tree": DecisionTreeModel(criterion="entropy"),
        "K-Nearest Neighbor": KNNModel(),
        "Naive Bayes (Gaussian)": NaiveBayesModel(),
        "Random Forest (Ensemble)": RandomForestModel(n_estimators=100),
    }
    for model in instances.values():
        model.train(_X_train, _y_train)
    return instances


def _plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title)
    plt.tight_layout()
    return fig


with st.sidebar:
    st.title(SIDEBAR_TITLE)
    st.caption(SIDEBAR_CAPTION)
    st.divider()

    st.subheader("Upload test data (CSV)")
    uploaded_file = st.file_uploader(
        "Upload your test_data.csv here",
        type=["csv"],
        label_visibility="collapsed",
    )
    st.caption(UPLOAD_CAPTION)

    st.divider()

    st.subheader("Model")
    selected_model = st.selectbox(
        "Select a model",
        options=MODEL_NAMES,
        label_visibility="collapsed",
    )

st.title(PAGE_HEADING)

with st.spinner("Loading dataset…"):
    loader, X_train, X_test, y_train, y_test = _load_data()
    n_classes = len(loader.class_names)
    class_names = loader.class_names

with st.spinner("Training models…"):
    model_instances = _train_all_models(X_train, y_train, n_classes)

X_eval, y_eval = X_test, y_test
eval_label = "held-out test set (20 %)"

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        X_up, y_up = loader.preprocess_uploaded(uploaded_df)
        if len(y_up) > 0:
            X_eval, y_eval = X_up, y_up
            eval_label = f"uploaded CSV ({len(y_up)} instances)"
        else:
            st.sidebar.warning(UPLOAD_NO_TARGET_WARNING)
    except Exception as exc:
        st.sidebar.error(f"Could not parse uploaded CSV: {exc}")

model = model_instances[selected_model]
y_pred = model.predict(X_eval)
y_prob = model.predict_proba(X_eval)
result = EvaluationMetrics(n_classes=n_classes).compute(
    selected_model, y_eval, y_pred, y_prob, class_names=class_names
)

st.caption(f"{selected_model} · {eval_label}")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Accuracy", f"{result.accuracy:.4f}")
m2.metric("AUC", f"{result.auc:.4f}")
m3.metric("Precision", f"{result.precision:.4f}")
m4.metric("Recall", f"{result.recall:.4f}")
m5.metric("F1 Score", f"{result.f1:.4f}")
m6.metric("MCC", f"{result.mcc:.4f}")

st.divider()

col_cm, col_cr = st.columns(2)
with col_cm:
    st.subheader("Confusion matrix")
    st.markdown(CONFUSION_MATRIX_CAPTION)
    st.pyplot(
        _plot_confusion_matrix(
            result.confusion_mat,
            class_names,
            f"Confusion Matrix — {selected_model}",
        )
    )
with col_cr:
    st.subheader("Classification report")
    st.code(result.classification_rep, language=None)

st.subheader("Predictions")
pred_names = [class_names[int(i)] for i in y_pred]
true_names = [class_names[int(i)] for i in y_eval]
st.dataframe(
    pd.DataFrame({"actual": true_names, "predicted": pred_names}),
    width=500,
    height=280,
)
