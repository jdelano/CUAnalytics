"""
Topic modeling with Latent Dirichlet Allocation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation

from .vectorizer import TextVectorizer


class LatentDirichletAllocationModel:
    """
    Student-friendly wrapper around scikit-learn's LatentDirichletAllocation.
    """

    def __init__(
        self,
        df,
        text_col,
        n_topics=5,
        max_iter=10,
        learning_method="batch",
        random_state=None,
        doc_topic_prior=None,
        topic_word_prior=None,
        max_features=None,
        min_df=1,
        max_df=1.0,
        max_n=1,
        lowercase=True,
        remove_stopwords=True,
        stem=False,
        keep_numbers=False,
    ):
        self.original_df = df
        self.text_col = text_col
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.learning_method = learning_method
        self.random_state = random_state
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.max_n = max_n
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.keep_numbers = keep_numbers

        self.vectorizer = None
        self.model = None
        self.feature_names = None
        self.topic_columns_ = None
        self._train_counts = None
        self._train_topic_matrix = None
        self._train_texts = None
        self._train_index = None

        self._validate_params()
        self._validate_input_df(df)
        self._fit(df)
        self._print_fit_summary()

    def _validate_params(self):
        if not isinstance(self.n_topics, int) or self.n_topics < 1:
            raise ValueError("n_topics must be an integer >= 1")
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("max_iter must be an integer >= 1")
        allowed_learning_methods = {"batch", "online"}
        if self.learning_method not in allowed_learning_methods:
            raise ValueError(
                f"learning_method must be one of {sorted(allowed_learning_methods)}"
            )

    def _validate_input_df(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if self.text_col not in df.columns:
            raise ValueError(f"Text column '{self.text_col}' not found in data")

    def _fit(self, df):
        self.vectorizer = TextVectorizer(
            text_col=self.text_col,
            method="count",
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            max_n=self.max_n,
            lowercase=self.lowercase,
            remove_stopwords=self.remove_stopwords,
            stem=self.stem,
            keep_numbers=self.keep_numbers,
        ).fit(df)

        texts = df[self.text_col].fillna("").astype(str)
        counts = self.vectorizer.vectorizer.transform(texts)

        self.model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=self.max_iter,
            learning_method=self.learning_method,
            random_state=self.random_state,
            doc_topic_prior=self.doc_topic_prior,
            topic_word_prior=self.topic_word_prior,
        )
        self.model.fit(counts)

        self.feature_names = list(self.vectorizer.get_feature_names())
        self.topic_columns_ = [f"topic_{i}" for i in range(self.n_topics)]
        self._train_counts = counts
        self._train_topic_matrix = self.model.transform(counts)
        self._train_texts = texts.reset_index(drop=True).copy()
        self._train_index = df.index.copy()

    def _check_fitted(self):
        if self.model is None or self.vectorizer is None:
            raise RuntimeError(
                "Model has not been fitted yet. "
                "Create model with: model = fit_topic_lda(df, text_col='review')"
            )

    def _print_fit_summary(self):
        print("\nTopic LDA fitted successfully!")
        print(f"  Topics: {self.n_topics}")
        print(f"  Vocabulary size: {len(self.feature_names)}")
        print(f"  Training documents: {len(self._train_index)}")
        print(f"  Max iterations: {self.max_iter}")
        print(f"  Learning method: {self.learning_method}")

    def _count_matrix_to_topic_df(self, matrix, index):
        topic_matrix = self.model.transform(matrix)
        return pd.DataFrame(topic_matrix, index=index, columns=self.topic_columns_)

    def _transform_counts_from_texts(self, texts):
        series = pd.Series(list(texts), dtype="object").fillna("").astype(str)
        return self.vectorizer.vectorizer.transform(series)

    def _topic_prevalence(self, topic_df):
        return topic_df.mean(axis=0)

    def _topic_term_probabilities(self, topic):
        weights = self.model.components_[topic]
        return weights / weights.sum()

    def _format_topic_terms(self, top_terms_df):
        return ", ".join(
            f"{row.term} ({row.probability:.1%})"
            for row in top_terms_df.itertuples(index=False)
        )

    def _topic_name(self, topic, top_n=3):
        top_terms_df = self.top_terms(topic, top_n=top_n)
        return " / ".join(top_terms_df["term"].tolist())

    def predict(self, df):
        self._check_fitted()
        topic_df = self.transform(df)
        dominant = topic_df.to_numpy().argmax(axis=1)
        return pd.DataFrame(
            {
                "topic": dominant,
                "topic_name": [self._topic_name(int(topic)) for topic in dominant],
            },
            index=df.index,
        )

    def predict_proba(self, df):
        self._check_fitted()
        return self.transform(df)

    def transform(self, df):
        self._check_fitted()
        self._validate_input_df(df)
        counts = self._transform_counts_from_texts(df[self.text_col].fillna("").astype(str))
        return self._count_matrix_to_topic_df(counts, index=df.index)

    def transform_text(self, texts):
        self._check_fitted()
        if isinstance(texts, str):
            texts = [texts]
        index = pd.RangeIndex(start=0, stop=len(texts))
        counts = self._transform_counts_from_texts(texts)
        return self._count_matrix_to_topic_df(counts, index=index)

    def transform_query(self, text):
        self._check_fitted()
        return self.transform_text([text])

    def top_terms(self, topic, top_n=10):
        self._check_fitted()
        if not isinstance(topic, int) or topic < 0 or topic >= self.n_topics:
            raise ValueError(f"topic must be an integer between 0 and {self.n_topics - 1}")

        weights = self.model.components_[topic]
        probabilities = self._topic_term_probabilities(topic)
        order = np.argsort(probabilities)[::-1][:top_n]
        return pd.DataFrame(
            {
                "topic": topic,
                "term": [self.feature_names[i] for i in order],
                "weight": weights[order],
                "probability": probabilities[order],
            }
        ).reset_index(drop=True)

    def top_terms_by_topic(self, top_n=10):
        self._check_fitted()
        rows = []
        for topic in range(self.n_topics):
            topic_terms = self.top_terms(topic, top_n=top_n)
            topic_terms.insert(1, "rank", np.arange(1, len(topic_terms) + 1))
            rows.append(topic_terms)
        return pd.concat(rows, ignore_index=True)

    def describe_topics(self, top_n=10):
        self._check_fitted()
        topic_df = pd.DataFrame(
            self._train_topic_matrix,
            index=self._train_index,
            columns=self.topic_columns_,
        )
        prevalence = self._topic_prevalence(topic_df)
        dominant = topic_df.to_numpy().argmax(axis=1)
        dominant_counts = pd.Series(dominant).value_counts().sort_index()

        rows = []
        for topic in range(self.n_topics):
            top_terms_df = self.top_terms(topic, top_n=top_n)
            rows.append(
                {
                    "topic": topic,
                    "topic_name": self._topic_name(topic),
                    "top_terms": self._format_topic_terms(top_terms_df),
                    "prevalence": float(prevalence[f"topic_{topic}"]),
                    "dominant_documents": int(dominant_counts.get(topic, 0)),
                }
            )
        return pd.DataFrame(rows)

    def score(self, df=None):
        self._check_fitted()
        if df is None:
            counts = self._train_counts
            index = self._train_index
        else:
            self._validate_input_df(df)
            counts = self._transform_counts_from_texts(df[self.text_col].fillna("").astype(str))
            index = df.index

        topic_df = self._count_matrix_to_topic_df(counts, index=index)
        dominant = topic_df.to_numpy().argmax(axis=1)
        dominant_counts = (
            pd.Series(dominant)
            .value_counts()
            .sort_index()
            .reindex(range(self.n_topics), fill_value=0)
        )

        return {
            "log_likelihood": float(self.model.score(counts)),
            "perplexity": float(self.model.perplexity(counts)),
            "n_documents": int(topic_df.shape[0]),
            "dominant_topic_counts": {int(k): int(v) for k, v in dominant_counts.items()},
        }

    def get_metrics(self):
        self._check_fitted()
        topic_df = pd.DataFrame(
            self._train_topic_matrix,
            index=self._train_index,
            columns=self.topic_columns_,
        )
        prevalence = self._topic_prevalence(topic_df)
        score = self.score()

        return {
            "model_type": "topic_lda",
            "n_topics": self.n_topics,
            "n_documents": int(topic_df.shape[0]),
            "vocabulary_size": len(self.feature_names),
            "feature_names": self.feature_names,
            "score": score,
            "topic_prevalence": {col: float(val) for col, val in prevalence.items()},
            "top_terms_by_topic": self.describe_topics(top_n=10).to_dict(orient="records"),
        }

    def visualize(self, figsize=(10, 6), top_n=5):
        self._check_fitted()
        topic_summary = self.describe_topics(top_n=top_n)

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(topic_summary["topic"].astype(str), topic_summary["prevalence"])
        ax.set_title("Topic Prevalence")
        ax.set_xlabel("Topic")
        ax.set_ylabel("Average document proportion")
        ax.set_ylim(0, max(1.0, float(topic_summary["prevalence"].max()) * 1.15))

        for _, row in topic_summary.iterrows():
            ax.text(
                row["topic"],
                row["prevalence"] + 0.01,
                row["top_terms"],
                ha="center",
                va="bottom",
                rotation=30,
                fontsize=9,
            )

        plt.tight_layout()
        plt.show()

    def summary(self):
        self._check_fitted()
        report = self.score()
        print("\nTopic LDA fitted successfully!")
        print(f"  Topics: {self.n_topics}")
        print(f"  Vocabulary size: {len(self.feature_names)}")
        print(f"  Training documents: {report['n_documents']}")
        print(f"  Perplexity: {report['perplexity']:.4f}")
        print(f"  Log likelihood: {report['log_likelihood']:.4f}")
        print(f"  Max document frequency: {self.max_df}")

        topic_descriptions = self.describe_topics(top_n=5)
        print("\nTop terms by topic:")
        for _, row in topic_descriptions.iterrows():
            print(f"  Topic {row['topic']}: {row['top_terms']}")


def fit_topic_lda(
    df,
    text_col,
    n_topics=5,
    max_iter=10,
    learning_method="batch",
    random_state=None,
    doc_topic_prior=None,
    topic_word_prior=None,
    max_features=None,
    min_df=1,
    max_df=1.0,
    max_n=1,
    lowercase=True,
    remove_stopwords=True,
    stem=False,
    keep_numbers=False,
):
    """
    Fit and return a LatentDirichletAllocationModel.
    """
    return LatentDirichletAllocationModel(
        df=df,
        text_col=text_col,
        n_topics=n_topics,
        max_iter=max_iter,
        learning_method=learning_method,
        random_state=random_state,
        doc_topic_prior=doc_topic_prior,
        topic_word_prior=topic_word_prior,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        max_n=max_n,
        lowercase=lowercase,
        remove_stopwords=remove_stopwords,
        stem=stem,
        keep_numbers=keep_numbers,
    )
