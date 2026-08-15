import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from config import (
    CONFUSION_MATRIX_CAPTION,
    DATASET_FEATURE_COUNT,
    DATASET_OVERVIEW_MD,
    DATASET_TOTAL_INSTANCES,
    METRIC_COLS,
    MODEL_NAMES,
    PAGE_HEADING,
    PAGE_ICON,
    PAGE_SUBTITLE,
    PAGE_TITLE,
    SIDEBAR_CAPTION,
    SIDEBAR_INFO,
    SIDEBAR_TITLE,
    UPLOAD_CAPTION,
    UPLOAD_NO_TARGET_WARNING,
    UPLOAD_PROMPT,
    model_note,
)
from data import FetalHealthDataLoader
from evaluation import EvaluationMetrics, MetricsResult
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
    ax.set_xlabel("Predicted class", fontsize=11)
    ax.set_ylabel("Actual class", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def _evaluate_all(model_instances, X, y, n_classes, class_names):
    evaluator = EvaluationMetrics(n_classes=n_classes)
    results = {}
    for name, model in model_instances.items():
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
        results[name] = evaluator.compute(name, y, y_pred, y_prob, class_names=class_names)
    return results


def _metrics_df(results: dict[str, MetricsResult]) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        row = {"ML Model": name}
        row.update(r.to_dict())
        rows.append(row)
    return pd.DataFrame(rows).set_index("ML Model")


def _highlight_best(df: pd.DataFrame):
    return df.style.highlight_max(
        subset=METRIC_COLS,
        color="#d4f1c7",
        axis=0,
    ).format("{:.4f}")


def _weakest_class_recall(cm: np.ndarray, names: list[str]) -> tuple[str, float]:
    row_sums = cm.sum(axis=1).astype(float)
    recalls = np.divide(
        np.diag(cm).astype(float),
        row_sums,
        out=np.zeros_like(row_sums),
        where=row_sums > 0,
    )
    idx = int(np.argmin(recalls))
    return names[idx], float(recalls[idx])


def _dynamic_observations(
    comp_df: pd.DataFrame,
    results: dict[str, MetricsResult],
    names: list[str],
    knn_k: int | None,
    eval_label: str,
) -> dict[str, str]:
    f1_rank = comp_df["F1 Score"].rank(ascending=False, method="min").astype(int)
    n_models = len(comp_df)
    notes = {}
    for model_name in MODEL_NAMES:
        r = results[model_name]
        wins = [m for m in METRIC_COLS if comp_df[m].idxmax() == model_name]
        weak, weak_rec = _weakest_class_recall(r.confusion_mat, names)
        win_txt = (
            f" Leads this evaluation on {', '.join(wins)}."
            if wins
            else " Does not lead any of the six metrics on this evaluation."
        )
        knn_txt = f" Neighbours used: K = {knn_k}." if model_name == "K-Nearest Neighbor" else ""
        notes[model_name] = (
            f"On {eval_label}: Accuracy {r.accuracy:.4f}, AUC {r.auc:.4f}, "
            f"Precision {r.precision:.4f}, Recall {r.recall:.4f}, "
            f"F1 {r.f1:.4f}, MCC {r.mcc:.4f}. "
            f"F1 rank {int(f1_rank.loc[model_name])} of {n_models}."
            f"{win_txt} Lowest class recall is {weak} ({weak_rec:.2f})."
            f"{knn_txt}"
        )
    return notes


with st.sidebar:
    st.title(SIDEBAR_TITLE)
    st.caption(SIDEBAR_CAPTION)
    st.divider()

    st.subheader(":material/upload_file: Upload test data (CSV)")
    uploaded_file = st.file_uploader(
        "Upload your test_data.csv here",
        type=["csv"],
        label_visibility="collapsed",
    )
    st.caption(UPLOAD_CAPTION)

    st.divider()

    st.subheader(":material/tune: Model selection")
    selected_model = st.selectbox(
        "Select a model to inspect",
        options=MODEL_NAMES,
        label_visibility="collapsed",
    )

    st.divider()
    st.info(SIDEBAR_INFO)

st.title(PAGE_HEADING)
st.markdown(PAGE_SUBTITLE)
st.divider()

with st.spinner("Loading dataset…"):
    loader, X_train, X_test, y_train, y_test = _load_data()
    n_classes = len(loader.class_names)
    class_names = loader.class_names

with st.spinner("Training models…"):
    model_instances = _train_all_models(X_train, y_train, n_classes)

X_eval, y_eval = None, None
eval_label = ""

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

if X_eval is None:
    st.info(UPLOAD_PROMPT)
    st.stop()

all_results = _evaluate_all(model_instances, X_eval, y_eval, n_classes, class_names)
y_pred = model_instances[selected_model].predict(X_eval)

tab_overview, tab_selected, tab_compare = st.tabs(
    [
        ":material/bar_chart: Overview",
        f":material/analytics: {selected_model}",
        ":material/compare_arrows: All models comparison",
    ]
)

with tab_overview:
    st.subheader("Dataset overview")

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    info_col1.metric("Total instances", DATASET_TOTAL_INSTANCES)
    info_col2.metric("Features", DATASET_FEATURE_COUNT)
    info_col3.metric("Train split", f"{len(X_train):,}")
    info_col4.metric("Test split", f"{len(X_test):,}")

    st.markdown(DATASET_OVERVIEW_MD)

with tab_selected:
    result: MetricsResult = all_results[selected_model]

    st.subheader(f":material/analytics: {selected_model} — detailed results")
    st.caption(f"Evaluated on: {eval_label}")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{result.accuracy:.4f}")
    m2.metric("AUC Score", f"{result.auc:.4f}")
    m3.metric("Precision", f"{result.precision:.4f}")
    m4.metric("Recall", f"{result.recall:.4f}")
    m5.metric("F1 Score", f"{result.f1:.4f}")
    m6.metric("MCC", f"{result.mcc:.4f}")

    st.divider()

    col_cm, col_cr = st.columns([1, 1])

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

    st.divider()

    st.subheader("Model description")
    st.markdown(
        model_note(selected_model, model_instances["K-Nearest Neighbor"].k_value)
    )

    st.subheader("Predictions")
    pred_names = [class_names[int(i)] for i in y_pred]
    true_names = [class_names[int(i)] for i in y_eval]
    st.dataframe(
        pd.DataFrame({"actual": true_names, "predicted": pred_names}),
        width=500,
        height=280,
    )

with tab_compare:
    st.subheader("All models — evaluation metrics comparison table")
    st.caption(f"Evaluated on: {eval_label}")

    comp_df = _metrics_df(all_results)
    st.dataframe(_highlight_best(comp_df), width=900)

    st.subheader("Visual comparison")

    fig2, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    palette = sns.color_palette("Set2", len(comp_df))

    for idx, metric in enumerate(METRIC_COLS):
        ax = axes[idx]
        vals = comp_df[metric].values
        bars = ax.bar(comp_df.index, vals, color=palette)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(len(comp_df.index)))
        ax.set_xticklabels(comp_df.index, rotation=30, ha="right", fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig2)

    st.subheader("Model performance observations")
    st.caption("Generated from the current evaluation numbers, not fixed text.")

    observations = _dynamic_observations(
        comp_df,
        all_results,
        class_names,
        model_instances["K-Nearest Neighbor"].k_value,
        eval_label,
    )
    winner = comp_df["F1 Score"].idxmax()
    metric_leads = [m for m in METRIC_COLS if comp_df[m].idxmax() == winner]
    f1_rank = comp_df["F1 Score"].rank(ascending=False, method="min").astype(int)

    for model_name, obs in observations.items():
        rank = int(f1_rank.loc[model_name])
        with st.expander(
            f":material/info: {model_name} (F1 Rank {rank})",
            expanded=False,
        ):
            st.write(obs)

    st.success(
        f":material/emoji_events: **Overall winner on {eval_label}: {winner}**  \n"
        f"Highest F1 Score = **{comp_df.loc[winner, 'F1 Score']:.4f}**. "
        f"Also leads: {', '.join(metric_leads)}."
    )

    st.subheader("Confusion matrices — all models")
    cm_cols = st.columns(len(MODEL_NAMES))
    for col, name in zip(cm_cols, MODEL_NAMES):
        with col:
            st.markdown(f"**{name}**")
            fig_cm = _plot_confusion_matrix(
                all_results[name].confusion_mat,
                class_names,
                "",
            )
            st.pyplot(fig_cm)
