"""
Text vectorization utilities for educational text mining workflows.
"""

import re

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class TextVectorizer:
    """
    Fit and apply a bag-of-words style text representation.

    Parameters
    ----------
    text_col : str
        Column containing raw text.
    target_col : str | None
        Optional label column to append during transform().
    method : str
        'binary', 'count', 'tf', or 'tfidf'.
    max_features : int | None
        Maximum number of terms/ngrams to keep.
    min_df : int | float
        Minimum document frequency threshold.
    max_df : int | float
        Maximum document frequency threshold.
    max_n : int
        Include n-grams from size 1 through max_n.
    lowercase : bool
        Convert text to lowercase before tokenization.
    remove_stopwords : bool
        Remove common English stopwords.
    stem : bool
        Apply a lightweight educational stemmer.
    keep_numbers : bool
        If False, discard tokens that contain digits.
    """

    def __init__(
        self,
        text_col,
        target_col=None,
        method="tfidf",
        max_features=None,
        min_df=1,
        max_df=1.0,
        max_n=1,
        lowercase=True,
        remove_stopwords=True,
        stem=False,
        keep_numbers=False,
    ):
        self.text_col = text_col
        self.target_col = target_col
        self.method = method
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.max_n = max_n
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.keep_numbers = keep_numbers

        self.vectorizer = None
        self.feature_names_ = None
        self.output_columns_ = None
        self.vocabulary_ = None
        self._corpus_matrix = None
        self._fitted_index = None
        self._fitted_texts = None

    def _validate_params(self):
        allowed_methods = {"binary", "count", "tf", "tfidf"}
        if self.method not in allowed_methods:
            raise ValueError(f"method must be one of {sorted(allowed_methods)}")
        if not isinstance(self.max_n, int) or self.max_n < 1:
            raise ValueError("max_n must be an integer >= 1")

    def _validate_input_df(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if self.text_col not in df.columns:
            raise ValueError(f"Text column '{self.text_col}' not found in data")

    def _normalize_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text)
        if self.lowercase:
            text = text.lower()
        return text

    def _simple_stem(self, token):
        """
        Lightweight suffix stripping for educational use.
        """
        replacements = [
            ("ingly", ""),
            ("edly", ""),
            ("ing", ""),
            ("edly", ""),
            ("ed", ""),
            ("ies", "y"),
            ("sses", "ss"),
            ("s", ""),
        ]
        for suffix, replacement in replacements:
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                candidate = token[: -len(suffix)] + replacement
                if len(candidate) >= 3:
                    return candidate
        return token

    def _tokenize(self, doc):
        doc = self._normalize_text(doc)
        tokens = re.findall(r"\b\w+\b", doc)
        if not self.keep_numbers:
            tokens = [token for token in tokens if not any(ch.isdigit() for ch in token)]
        tokens = [
            token for token in tokens
            if len(token) > 1 or any(ch.isdigit() for ch in token)
        ]
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
        if self.stem:
            tokens = [self._simple_stem(token) for token in tokens]
        return [token for token in tokens if token]

    def _create_vectorizer(self):
        common_kwargs = {
            "preprocessor": self._normalize_text,
            "tokenizer": self._tokenize,
            "token_pattern": None,
            "ngram_range": (1, self.max_n),
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
        }
        if self.method == "tfidf":
            return TfidfVectorizer(norm="l2", **common_kwargs)
        if self.method == "binary":
            return CountVectorizer(binary=True, **common_kwargs)
        return CountVectorizer(binary=False, **common_kwargs)

    def _build_output_columns(self, feature_names):
        used = set()
        output_columns = []
        for feature in feature_names:
            safe = re.sub(r"[^0-9A-Za-z_]+", "_", feature).strip("_")
            if not safe:
                safe = "term"
            safe = f"term__{safe}"
            candidate = safe
            counter = 2
            while candidate in used:
                candidate = f"{safe}_{counter}"
                counter += 1
            used.add(candidate)
            output_columns.append(candidate)
        return output_columns

    def fit(self, df):
        """
        Learn vocabulary and weighting statistics from a training DataFrame.
        """
        self._validate_params()
        self._validate_input_df(df)

        self.vectorizer = self._create_vectorizer()
        texts = df[self.text_col].fillna("").astype(str)
        matrix = self.vectorizer.fit_transform(texts)
        if self.method == "tf":
            matrix = normalize(matrix, norm="l1", axis=1)

        self.feature_names_ = list(self.vectorizer.get_feature_names_out())
        self.output_columns_ = self._build_output_columns(self.feature_names_)
        self.vocabulary_ = dict(zip(self.feature_names_, self.output_columns_))
        self._corpus_matrix = matrix
        self._fitted_index = df.index.copy()
        self._fitted_texts = texts.reset_index(drop=True).copy()
        return self

    def _check_fitted(self):
        if self.vectorizer is None or self.feature_names_ is None:
            raise RuntimeError(
                "TextVectorizer has not been fitted yet. "
                "Create one with: vec = fit_text_vectorizer(df, text_col='review')"
            )

    def _matrix_to_dataframe(self, matrix, index):
        if issparse(matrix):
            matrix = matrix.toarray()
        return pd.DataFrame(matrix, index=index, columns=self.output_columns_)

    def transform_text(self, texts):
        """
        Transform raw text values into a feature DataFrame.
        """
        self._check_fitted()

        if isinstance(texts, str):
            texts = [texts]

        index = pd.RangeIndex(start=0, stop=len(texts))
        matrix = self.vectorizer.transform(pd.Series(list(texts), dtype="object"))
        if self.method == "tf":
            matrix = normalize(matrix, norm="l1", axis=1)
        return self._matrix_to_dataframe(matrix, index=index)

    def transform_query(self, text):
        """
        Transform a single query string into a one-row feature DataFrame.
        """
        return self.transform_text([text])

    def transform(self, df, include_target=True):
        """
        Transform a DataFrame containing raw text into text features.
        """
        self._validate_input_df(df)
        transformed = self.transform_text(df[self.text_col].fillna("").astype(str).tolist())
        transformed.index = df.index

        if include_target and self.target_col and self.target_col in df.columns:
            transformed[self.target_col] = df[self.target_col]
        return transformed

    def fit_transform(self, df, include_target=True):
        """
        Fit on df, then transform it.
        """
        self.fit(df)
        return self.transform(df, include_target=include_target)

    def get_feature_names(self):
        """
        Return the learned terms/ngrams in vocabulary order.
        """
        self._check_fitted()
        return list(self.feature_names_)

    def summary(self):
        """
        Print a short summary of the fitted vectorizer.
        """
        self._check_fitted()
        print("\nText Vectorizer fitted successfully!")
        print(f"  Text column: {self.text_col}")
        print(f"  Method: {self.method}")
        print(f"  Max n-gram size: {self.max_n}")
        print(f"  Vocabulary size: {len(self.feature_names_)}")
        print(f"  Stopwords removed: {self.remove_stopwords}")
        print(f"  Stemming enabled: {self.stem}")

    def top_terms(self, df=None, top_n=20, by="frequency"):
        """
        Return the most heavily weighted terms in the fitted corpus or a provided DataFrame.
        """
        self._check_fitted()
        if by not in {"frequency", "tfidf"}:
            raise ValueError("by must be 'frequency' or 'tfidf'")

        if df is None:
            texts = self._fitted_texts
            matrix = self._corpus_matrix
        else:
            texts = df[self.text_col].fillna("").astype(str)
            if by == "tfidf":
                matrix = self.vectorizer.transform(texts)
                if self.method == "tf":
                    matrix = normalize(matrix, norm="l1", axis=1)

        if by == "frequency":
            count_vectorizer = CountVectorizer(
                vocabulary=self.vectorizer.vocabulary_,
                preprocessor=self._normalize_text,
                tokenizer=self._tokenize,
                token_pattern=None,
                ngram_range=(1, self.max_n),
            )
            count_matrix = count_vectorizer.transform(texts)
            scores = np.asarray(count_matrix.sum(axis=0)).ravel()
        else:
            scores = np.asarray(matrix.sum(axis=0)).ravel()

        results = pd.DataFrame({"term": self.feature_names_, "score": scores})
        return results.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    def top_terms_by_class(self, df, top_n=20):
        """
        Rank terms by information gain with respect to the configured target column.
        """
        self._check_fitted()
        if self.target_col is None:
            raise ValueError("target_col must be set to compute class-based term rankings.")
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")

        transformed = self.transform(df, include_target=False)
        y = df[self.target_col]

        from sklearn.feature_selection import mutual_info_classif

        scores = mutual_info_classif(transformed.values, y)
        results = pd.DataFrame({"term": self.feature_names_, "information_gain": scores})
        return results.sort_values("information_gain", ascending=False).head(top_n).reset_index(drop=True)

    def similarity_search(self, df, query, top_n=5):
        """
        Return the most similar documents in df to a query string.
        """
        self._validate_input_df(df)
        document_features = self.transform(df, include_target=False)
        query_features = self.transform_query(query)
        scores = cosine_similarity(query_features.values, document_features.values).ravel()

        results = df.copy()
        results["similarity"] = scores
        return results.sort_values("similarity", ascending=False).head(top_n)


def fit_text_vectorizer(
    df,
    text_col,
    target_col=None,
    method="tfidf",
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
    Fit and return a TextVectorizer.
    """
    vectorizer = TextVectorizer(
        text_col=text_col,
        target_col=target_col,
        method=method,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        max_n=max_n,
        lowercase=lowercase,
        remove_stopwords=remove_stopwords,
        stem=stem,
        keep_numbers=keep_numbers,
    )
    return vectorizer.fit(df)
