import pandas as pd
import pytest

from cuanalytics import fit_kmeans, fit_logit, fit_text_vectorizer
from cuanalytics.text import TextVectorizer


@pytest.fixture
def movie_reviews():
    reviews = [
        ("A thrilling and moving film with excellent acting", "Hot"),
        ("Brilliant script and strong performances throughout", "Hot"),
        ("A warm funny story with memorable characters", "Hot"),
        ("Excellent direction and a gripping ending", "Hot"),
        ("The movie was inspiring, clever, and beautifully shot", "Hot"),
        ("An engaging story with great pacing and strong dialogue", "Hot"),
        ("A dull and boring movie with weak acting", "Not"),
        ("Terrible script, flat characters, and a bad ending", "Not"),
        ("Predictable plot and painfully slow pacing", "Not"),
        ("The acting was wooden and the story was lifeless", "Not"),
        ("Messy direction and an uninteresting cast", "Not"),
        ("A forgettable film with poor dialogue and weak visuals", "Not"),
    ]
    return pd.DataFrame(reviews, columns=["review", "label"])


def test_fit_text_vectorizer_returns_object(movie_reviews):
    vec = fit_text_vectorizer(movie_reviews, text_col="review", target_col="label")
    assert isinstance(vec, TextVectorizer)


def test_transform_includes_target_by_default(movie_reviews):
    vec = fit_text_vectorizer(movie_reviews, text_col="review", target_col="label", max_features=20)
    transformed = vec.transform(movie_reviews)
    assert "label" in transformed.columns
    assert transformed["label"].tolist() == movie_reviews["label"].tolist()


def test_transform_can_exclude_target(movie_reviews):
    vec = fit_text_vectorizer(movie_reviews, text_col="review", target_col="label", max_features=20)
    transformed = vec.transform(movie_reviews, include_target=False)
    assert "label" not in transformed.columns


def test_transform_query_returns_one_row(movie_reviews):
    vec = fit_text_vectorizer(movie_reviews, text_col="review", target_col="label", max_features=20)
    query = vec.transform_query("excellent acting and strong story")
    assert len(query) == 1
    assert "label" not in query.columns


def test_max_n_creates_trigrams(movie_reviews):
    vec = fit_text_vectorizer(
        movie_reviews,
        text_col="review",
        target_col="label",
        method="count",
        max_n=3,
        min_df=1,
    )
    feature_names = vec.get_feature_names()
    assert any(len(term.split()) == 3 for term in feature_names)


def test_logistic_regression_workflow(movie_reviews, monkeypatch):
    train = movie_reviews.iloc[:10].copy()
    test = movie_reviews.iloc[10:].copy()

    vec = fit_text_vectorizer(
        train,
        text_col="review",
        target_col="label",
        method="tfidf",
        max_features=30,
        max_n=3,
    )

    train_vec = vec.transform(train)
    test_vec = vec.transform(test)

    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)
    model = fit_logit(train_vec, formula="label ~ .")
    report = model.score(test_vec)

    assert set(model.classes) == {"Hot", "Not"}
    assert "accuracy" in report


def test_kmeans_workflow(movie_reviews, monkeypatch):
    vec = fit_text_vectorizer(
        movie_reviews,
        text_col="review",
        method="tfidf",
        max_features=25,
        max_n=3,
    )
    transformed = vec.transform(movie_reviews, include_target=False)

    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)
    model = fit_kmeans(transformed, formula=".", n_clusters=2, random_state=42)
    metrics = model.get_metrics()

    assert metrics["n_clusters"] == 2
    assert metrics["n_features"] <= 25


def test_similarity_search_returns_sorted_results(movie_reviews):
    vec = fit_text_vectorizer(
        movie_reviews,
        text_col="review",
        target_col="label",
        method="tfidf",
        max_features=30,
    )
    results = vec.similarity_search(movie_reviews, "excellent acting and brilliant script", top_n=3)

    assert len(results) == 3
    assert "similarity" in results.columns
    assert results["similarity"].is_monotonic_decreasing


def test_top_terms_frequency_uses_counts_not_tfidf_weights():
    df = pd.DataFrame(
        {
            "review": [
                "alpha alpha alpha beta",
                "alpha beta",
                "beta gamma",
            ]
        }
    )
    vec = fit_text_vectorizer(
        df,
        text_col="review",
        method="tfidf",
        max_features=None,
        remove_stopwords=False,
        stem=False,
    )

    frequency_terms = vec.top_terms(top_n=3, by="frequency")
    tfidf_terms = vec.top_terms(top_n=3, by="tfidf")

    assert frequency_terms.iloc[0]["term"] == "alpha"
    assert frequency_terms.iloc[0]["score"] == 4
    assert frequency_terms.iloc[1]["term"] == "beta"
    assert frequency_terms.iloc[1]["score"] == 3
    assert not frequency_terms["score"].equals(tfidf_terms["score"])


def test_top_terms_frequency_on_subset_uses_counts():
    df = pd.DataFrame(
        {
            "review": [
                "alpha alpha beta",
                "beta gamma",
                "gamma gamma",
            ]
        }
    )
    vec = fit_text_vectorizer(
        df,
        text_col="review",
        method="tfidf",
        max_features=None,
        remove_stopwords=False,
        stem=False,
    )

    subset = df.iloc[[1, 2]]
    terms = vec.top_terms(df=subset, top_n=3, by="frequency")

    assert terms.iloc[0]["term"] == "gamma"
    assert terms.iloc[0]["score"] == 3
    assert terms.iloc[1]["term"] == "beta"
    assert terms.iloc[1]["score"] == 1
