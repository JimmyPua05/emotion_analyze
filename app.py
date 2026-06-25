"""Streamlit app for Social Media Emotion Detection."""

from __future__ import annotations

from pathlib import Path
import textwrap

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_preprocessing import LABEL_NAMES, load_processed_dataset, preprocess_text
from src.debug_tools import FAIL, PASS, WARN, run_diagnostics, summarize_diagnostics
from src.predict import available_model_names, explain_prediction_terms, get_model_registry, model_artifact_exists, predict_emotion
from src.visualization import (
    add_text_length,
    load_model_details,
    load_model_results,
    top_words_by_emotion,
    unigram_bigram_summary,
)


st.set_page_config(page_title="Social Media Emotion Analyzer", page_icon=":bar_chart:", layout="wide")

TEAM_MEMBERS = ["Jimmy", "Arif", "Haziq"]


@st.cache_data
def cached_dataset() -> pd.DataFrame:
    """Load cleaned data once for the Streamlit session."""

    try:
        return load_processed_dataset()
    except Exception:
        return pd.DataFrame(columns=["text", "emotion", "clean_text"])


@st.cache_data
def cached_results() -> pd.DataFrame:
    """Load saved model comparison data once for the Streamlit session."""

    return load_model_results()


@st.cache_data
def cached_model_details() -> list[dict]:
    """Load detailed model metrics such as confusion matrices."""

    return load_model_details()


def create_horizontal_bar_figure(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
    wrap_x_labels: bool = False,
    height: int = 520,
    bottom_margin: int = 120,
):
    """Create a bar chart with horizontal category labels.

    Streamlit's built-in bar chart can rotate short category names vertically.
    Plotly gives us direct control, so emotion names stay horizontal under each
    bar and are easier to read during the demo.
    """

    fig = px.bar(df, x=x, y=y, color=color, text=y, title=title)
    fig.update_traces(textposition="outside", cliponaxis=False)
    tick_values = df[x].astype(str).tolist()
    tick_text = [
        "<br>".join(textwrap.wrap(value, width=18, break_long_words=False))
        if wrap_x_labels
        else value
        for value in tick_values
    ]
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=tick_values,
        tickangle=0,
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_text,
        automargin=True,
        tickfont=dict(size=10 if wrap_x_labels else 12),
    )
    fig.update_layout(
        xaxis_title=x.replace("_", " ").title(),
        yaxis_title=y.replace("_", " ").title(),
        margin=dict(t=70, b=bottom_margin),
        height=height,
    )
    return fig


def show_horizontal_bar_chart(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
    wrap_x_labels: bool = False,
    height: int = 520,
    bottom_margin: int = 120,
) -> None:
    """Render a Plotly bar chart with readable horizontal category labels."""

    fig = create_horizontal_bar_figure(
        df,
        x=x,
        y=y,
        title=title,
        color=color,
        wrap_x_labels=wrap_x_labels,
        height=height,
        bottom_margin=bottom_margin,
    )
    st.plotly_chart(fig, use_container_width=True)


def create_model_comparison_figure(results: pd.DataFrame):
    """Create a readable horizontal chart for many long model names.

    A vertical bar chart is too cramped for 19 model combinations. Horizontal
    bars give every model name its own line and keep the F1 labels readable.
    """

    ranking = results.sort_values("f1", ascending=True).copy()
    ranking["f1_label"] = ranking["f1"].map(lambda value: f"{value:.3f}")
    model_names = ranking["display_name"].astype(str).tolist()
    height = max(720, 34 * len(ranking) + 180)
    hover_data = {"display_name": True, "f1": ":.3f"}
    for metric in ["accuracy", "precision", "recall"]:
        if metric in ranking.columns:
            hover_data[metric] = ":.3f"

    fig = px.bar(
        ranking,
        x="f1",
        y="display_name",
        orientation="h",
        text="f1_label",
        title="Model Comparison By F1-Score",
        hover_data=hover_data,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(
        type="category",
        tickmode="array",
        tickvals=model_names,
        ticktext=model_names,
        automargin=True,
        title=None,
    )
    fig.update_xaxes(
        title="Weighted F1-Score",
        range=[0, 1.05],
        tickformat=".2f",
    )
    fig.update_layout(
        height=height,
        margin=dict(l=340, r=80, t=70, b=60),
        bargap=0.22,
    )
    return fig


def show_home() -> None:
    st.title("Social Media Emotion Detection Using Machine Learning")
    st.write(
        "This app predicts emotions in social media-style text using classical "
        "machine learning models and a bonus DistilBERT transformer model."
    )

    st.subheader("Team Members")
    st.write(", ".join(TEAM_MEMBERS))

    st.subheader("Problem And Objective")
    st.write(
        "Social media posts can contain many different emotions. The objective "
        "is to classify a post into sadness, joy, love, anger, fear, or surprise "
        "and explain the result with confidence scores and visual insights."
    )

    st.subheader("How To Use The App")
    st.markdown(
        """
        1. Open **Text Analyzer**.
        2. Choose any trained model from the dropdown.
        3. Enter a social-media-style sentence.
        4. Click **Analyze Emotion** to see the predicted emotion, confidence scores, and influential words or phrases.
        5. Use **Data Explorer**, **Visualizations**, and **Model Info** to inspect the dataset and model performance.
        """
    )

    st.markdown(
        """
        **Emotion labels:** sadness, joy, love, anger, fear, surprise

        **Bonus work included:**
        - N-gram insights comparing unigram and bigram features
        - DistilBERT transformer model for advanced NLP
        - Multiple visualizations for dataset and model performance
        """
    )


def show_text_analyzer() -> None:
    st.title("Text Analyzer")
    registry = get_model_registry()
    model_names = available_model_names(registry)
    if not model_names:
        st.error("No trained model artifacts are available. Run the training scripts or restore the model files first.")
        return
    preferred_order = [
        "DistilBERT",
        "TF-IDF Bigram + SVM",
        "Count Bigram + SVM",
        "Count Bigram + Logistic Regression",
        "Count Unigram + Logistic Regression",
    ]
    ordered_model_names = [name for name in preferred_order if name in model_names] + [name for name in model_names if name not in preferred_order]
    model_name = st.selectbox("Choose a model", ordered_model_names, index=0)
    st.caption(registry[model_name].description)

    text = st.text_area(
        "Enter social media text",
        value="I feel so happy and excited about today",
        height=140,
    )

    cleaned = preprocess_text(text)
    st.write("Cleaned text:", cleaned if cleaned else "(empty after cleaning)")

    if st.button("Analyze Emotion", type="primary"):
        try:
            prediction, confidence = predict_emotion(model_name, text)
        except FileNotFoundError as exc:
            st.warning(str(exc))
            st.info("Run the training scripts first: `python -m src.train_classical` and `python -m src.train_transformer`.")
            return
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        st.subheader(f"Predicted Emotion: {prediction}")
        confidence_df = (
            pd.DataFrame({"emotion": list(confidence), "confidence": list(confidence.values())})
            .sort_values("confidence", ascending=False)
            .reset_index(drop=True)
        )
        show_horizontal_bar_chart(
            confidence_df,
            x="emotion",
            y="confidence",
            title="Prediction Confidence By Emotion",
        )
        st.dataframe(confidence_df, use_container_width=True)

        explanation_df = explain_prediction_terms(model_name, text)
        st.subheader("Words/Phrases Influencing Prediction")
        if explanation_df.empty:
            st.info("No readable influence terms were found for this input.")
        else:
            st.dataframe(explanation_df, use_container_width=True)


def show_data_explorer() -> None:
    st.title("Data Explorer")
    df = cached_dataset()
    if df.empty:
        st.warning("No processed dataset found. Run `python -m src.data_preprocessing` first.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Emotion Classes", df["emotion"].nunique())
    col3.metric("Average Text Length", round(add_text_length(df)["text_length"].mean(), 2))

    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Emotion Distribution")
    distribution = df["emotion"].value_counts().rename_axis("emotion").reset_index(name="count")
    show_horizontal_bar_chart(
        distribution,
        x="emotion",
        y="count",
        title="Emotion Distribution",
    )


def show_visualizations() -> None:
    st.title("Visualizations And Insights")
    df = cached_dataset()
    results = cached_results()

    if df.empty:
        st.warning("No dataset available yet. Run preprocessing first.")
        return

    st.subheader("Class Distribution")
    show_horizontal_bar_chart(
        df["emotion"].value_counts().rename_axis("emotion").reset_index(name="count"),
        x="emotion",
        y="count",
        title="Class Distribution",
    )

    st.subheader("Text Length Distribution By Emotion")
    length_df = add_text_length(df)
    st.scatter_chart(length_df, x="emotion", y="text_length")

    st.subheader("Top Words Per Emotion")
    emotion = st.selectbox("Choose emotion for top words", LABEL_NAMES)
    st.bar_chart(top_words_by_emotion(df, emotion), x="word", y="count")

    st.subheader("Word Cloud")
    try:
        from wordcloud import WordCloud

        words = " ".join(df[df["emotion"] == emotion]["clean_text"].fillna(""))
        cloud = WordCloud(width=900, height=420, background_color="white").generate(words)
        st.image(cloud.to_array(), caption=f"Word cloud for {emotion}")
    except Exception as exc:
        st.info(f"Install wordcloud to show this chart: {exc}")

    st.subheader("Unigram vs Bigram Insight")
    ngram_summary = unigram_bigram_summary(results)
    if ngram_summary.empty:
        st.info("Train classical models first to compare unigram and bigram performance.")
    else:
        show_horizontal_bar_chart(
            ngram_summary,
            x="ngram_type",
            y="f1",
            title="Unigram vs Bigram Average F1-Score",
        )
        st.write(
            "This insight helps explain whether two-word phrases such as "
            "`not happy` improve emotion detection compared with single words only."
        )

    st.subheader("Model Comparison")
    if results.empty:
        st.info("Train models first to show model comparison.")
    else:
        st.plotly_chart(create_model_comparison_figure(results), use_container_width=True)


def show_model_info() -> None:
    st.title("Model Info")
    results = cached_results()
    if results.empty:
        st.warning("No model results found. Run `python -m src.train_classical` first.")
        return

    st.subheader("Model Comparison")
    visible_cols = ["display_name", "feature_method", "classifier", "accuracy", "precision", "recall", "f1"]
    st.dataframe(results[visible_cols].sort_values("f1", ascending=False), use_container_width=True)

    st.subheader("Best Overall Model")
    best = results.sort_values("f1", ascending=False).iloc[0]
    st.success(f"{best['display_name']} has the highest weighted F1-score: {best['f1']:.3f}")

    st.subheader("Advanced NLP Bonus Model")
    distilbert_row = results[results["display_name"] == "DistilBERT"]
    if distilbert_row.empty:
        st.warning("DistilBERT was trained, but its evaluation metrics are not saved in the comparison file yet.")
    else:
        distilbert = distilbert_row.iloc[0]
        st.info(
            f"DistilBERT is included as the advanced NLP transformer model. "
            f"Its weighted F1-score is {distilbert['f1']:.3f}, so it appears in the same comparison table and chart as the classical models."
        )

    st.subheader("Confusion Matrix")
    details = cached_model_details()
    if not details:
        st.info("Train classical models first to show confusion matrices.")
    else:
        detail_by_name = {item["display_name"]: item for item in details}
        model_names = list(detail_by_name)
        preferred_order = [
            "DistilBERT",
            "TF-IDF Bigram + SVM",
            "Count Bigram + SVM",
            "Count Bigram + Logistic Regression",
            "Count Unigram + Logistic Regression",
        ]
        ordered_model_names = [name for name in preferred_order if name in model_names] + [name for name in model_names if name not in preferred_order]
        selected = st.selectbox("Choose model for confusion matrix", ordered_model_names)
        matrix = pd.DataFrame(
            detail_by_name[selected]["confusion_matrix"],
            index=LABEL_NAMES,
            columns=LABEL_NAMES,
        )
        try:
            fig = px.imshow(
                matrix,
                text_auto=True,
                labels={"x": "Predicted emotion", "y": "Actual emotion", "color": "Count"},
                title=f"Confusion Matrix: {selected}",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.dataframe(matrix, use_container_width=True)

    st.markdown(
        """
        **CountVectorizer:** counts words or phrases.

        **TF-IDF:** gives higher weight to words that are important but not too common.

        **Word2Vec:** represents words as dense meaning-based vectors.

        **DistilBERT:** uses a transformer architecture, so it understands word context better than simple word counts.
        """
    )




def show_diagnostics() -> None:
    """Show project health checks to make debugging easier for teammates."""

    st.title("Diagnostics")
    checks = run_diagnostics()
    summary = summarize_diagnostics(checks)

    col1, col2, col3 = st.columns(3)
    col1.metric("Passing Checks", summary.get(PASS, 0))
    col2.metric("Warnings", summary.get(WARN, 0))
    col3.metric("Failures", summary.get(FAIL, 0))

    checks_df = pd.DataFrame(checks)
    st.dataframe(checks_df, use_container_width=True)

    problem_df = checks_df[checks_df["status"].isin([WARN, FAIL])]
    if problem_df.empty:
        st.success("All required diagnostics passed. The project is ready to run.")
    else:
        st.subheader("What To Fix")
        st.dataframe(problem_df[["check", "status", "details", "fix"]], use_container_width=True)

    st.subheader("Useful Debug Commands")
    st.code(
        "python scripts/debug_project.py\n"
        "python -m unittest tests.test_core\n"
        "streamlit run app.py",
        language="bash",
    )

def main() -> None:
    page = st.sidebar.radio(
        "Navigation",
        ["Home/About", "Text Analyzer", "Data Explorer", "Visualizations", "Model Info", "Diagnostics"],
    )
    if page == "Home/About":
        show_home()
    elif page == "Text Analyzer":
        show_text_analyzer()
    elif page == "Data Explorer":
        show_data_explorer()
    elif page == "Visualizations":
        show_visualizations()
    elif page == "Model Info":
        show_model_info()
    else:
        show_diagnostics()


if __name__ == "__main__":
    main()




