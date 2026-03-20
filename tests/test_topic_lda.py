import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from cuanalytics import fit_topic_lda
from cuanalytics.text import LatentDirichletAllocationModel


@pytest.fixture
def themed_documents():
    docs = [
        "team win game season coach player score",
        "basketball team offense defense player season",
        "soccer match goal team coach league",
        "player training game tournament team",
        "recipe kitchen dinner flavor cook ingredients",
        "bake bread kitchen recipe meal ingredients",
        "chef prepares dinner with fresh flavor herbs",
        "cooking meal recipe pan sauce kitchen",
    ]
    return pd.DataFrame({"text": docs})


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


def test_fit_topic_lda_returns_object(themed_documents):
    model = fit_topic_lda(themed_documents, text_col="text", n_topics=2, random_state=42)
    assert isinstance(model, LatentDirichletAllocationModel)


def test_transform_returns_topic_dataframe(themed_documents):
    model = fit_topic_lda(themed_documents, text_col="text", n_topics=2, random_state=42)
    transformed = model.transform(themed_documents)

    assert list(transformed.columns) == ["topic_0", "topic_1"]
    assert transformed.shape == (len(themed_documents), 2)
    assert all(abs(row_sum - 1.0) < 1e-6 for row_sum in transformed.sum(axis=1))


def test_predict_and_predict_proba_are_consistent(themed_documents):
    model = fit_topic_lda(themed_documents, text_col="text", n_topics=2, random_state=42)

    probs = model.predict_proba(themed_documents)
    labels = model.predict(themed_documents)

    assert probs.shape == (len(themed_documents), 2)
    assert list(labels.columns) == ["topic", "topic_name"]
    assert set(labels["topic"].unique()).issubset({0, 1})
    assert labels["topic_name"].str.len().gt(0).all()
    assert (probs.to_numpy().argmax(axis=1) == labels["topic"].to_numpy()).all()


def test_transform_text_and_query(themed_documents):
    model = fit_topic_lda(themed_documents, text_col="text", n_topics=2, random_state=42)

    transformed = model.transform_text(["team game coach", "recipe dinner kitchen"])
    query = model.transform_query("player season tournament")

    assert transformed.shape == (2, 2)
    assert query.shape == (1, 2)


def test_top_terms_methods(themed_documents):
    model = fit_topic_lda(
        themed_documents,
        text_col="text",
        n_topics=2,
        random_state=42,
        remove_stopwords=False,
    )

    topic_terms = model.top_terms(0, top_n=3)
    all_topics = model.top_terms_by_topic(top_n=3)

    assert list(topic_terms.columns) == ["topic", "term", "weight", "probability"]
    assert len(topic_terms) == 3
    assert {"topic", "rank", "term", "weight", "probability"} == set(all_topics.columns)
    assert len(all_topics) == 6
    assert topic_terms["probability"].between(0, 1).all()
    assert topic_terms["probability"].is_monotonic_decreasing


def test_describe_topics_returns_readable_summary(themed_documents):
    model = fit_topic_lda(themed_documents, text_col="text", n_topics=2, random_state=42)
    summary = model.describe_topics(top_n=4)

    assert list(summary.columns) == ["topic", "topic_name", "top_terms", "prevalence", "dominant_documents"]
    assert len(summary) == 2
    assert summary["dominant_documents"].sum() == len(themed_documents)
    assert "%" in summary.iloc[0]["top_terms"]
    assert summary["topic_name"].str.len().gt(0).all()


def test_score_returns_fit_metrics(themed_documents):
    model = fit_topic_lda(themed_documents, text_col="text", n_topics=2, random_state=42)
    score = model.score()

    assert set(score.keys()) == {
        "log_likelihood",
        "perplexity",
        "n_documents",
        "dominant_topic_counts",
    }
    assert score["n_documents"] == len(themed_documents)
    assert sum(score["dominant_topic_counts"].values()) == len(themed_documents)
    assert score["perplexity"] > 0


def test_get_metrics_includes_topic_metadata(themed_documents):
    model = fit_topic_lda(themed_documents, text_col="text", n_topics=2, random_state=42)
    metrics = model.get_metrics()

    assert metrics["model_type"] == "topic_lda"
    assert metrics["n_topics"] == 2
    assert metrics["n_documents"] == len(themed_documents)
    assert "score" in metrics
    assert "topic_prevalence" in metrics
    assert len(metrics["top_terms_by_topic"]) == 2


def test_visualize_runs_without_error(themed_documents, monkeypatch):
    model = fit_topic_lda(themed_documents, text_col="text", n_topics=2, random_state=42)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    model.visualize()


def test_topic_lda_respects_max_df():
    df = pd.DataFrame(
        {
            "text": [
                "movie hero battle",
                "movie drama family",
                "movie mystery night",
                "movie comedy romance",
            ]
        }
    )

    model = fit_topic_lda(
        df,
        text_col="text",
        n_topics=2,
        random_state=42,
        remove_stopwords=False,
        max_df=0.75,
    )

    assert "movie" not in model.feature_names
