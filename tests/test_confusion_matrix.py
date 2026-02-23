import numpy as np
import pandas as pd
import pytest

from cuanalytics import (
    ConfusionMatrix,
    fit_knn_classifier,
    fit_logit,
    fit_nn,
    fit_svm,
    fit_tree,
    fit_lda,
)


def test_confusion_matrix_binary_metrics():
    y_true = ['N', 'N', 'N', 'P', 'P', 'P']
    y_pred = ['N', 'P', 'N', 'P', 'N', 'P']

    cm = ConfusionMatrix(y_true, y_pred, labels=['N', 'P'])
    metrics = cm.get_metrics()

    assert cm.matrix.tolist() == [[2, 1], [1, 2]]
    assert metrics['binary_counts'] == {'tp': 2, 'fp': 1, 'fn': 1, 'tn': 2}
    assert metrics['precision'] == pytest.approx(2 / 3)
    assert metrics['recall'] == pytest.approx(2 / 3)
    assert metrics['specificity'] == pytest.approx(2 / 3)
    assert metrics['fpr'] == pytest.approx(1 / 3)
    assert metrics['f_measure'] == pytest.approx(2 / 3)


def test_confusion_matrix_display_modes():
    y_true = ['N', 'N', 'P', 'P']
    y_pred = ['N', 'P', 'N', 'P']
    cm = ConfusionMatrix(y_true, y_pred, labels=['N', 'P'])

    normal = cm.to_dataframe(display='normal')
    inverted = cm.to_dataframe(display='inverted')

    assert normal.index.tolist() == ['Actual N', 'Actual P']
    assert normal.columns.tolist() == ['Pred N', 'Pred P']
    assert inverted.index.tolist() == ['Pred P', 'Pred N']
    assert inverted.columns.tolist() == ['Actual P', 'Actual N']


def test_confusion_matrix_inverted_uses_textbook_binary_layout():
    y_true = ['No', 'No', 'No', 'No', 'Yes', 'Yes']
    y_pred = ['No', 'Yes', 'Yes', 'No', 'Yes', 'No']
    cm = ConfusionMatrix(y_true, y_pred, labels=['No', 'Yes'])

    inverted = cm.to_dataframe(display='inverted')
    assert inverted.values.tolist() == [
        [1, 2],  # TP, FP
        [1, 2],  # FN, TN
    ]


def test_confusion_matrix_expected_value():
    # Normal matrix (Actual rows, Pred columns) = [[4, 1], [2, 3]]
    # Inverted display matrix (Pred rows, Actual columns) = [[3, 1], [2, 4]]
    y_true = ['No', 'No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes']
    y_pred = ['No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes']
    cm = ConfusionMatrix(y_true, y_pred, labels=['No', 'Yes'])

    # Costs follow inverted binary layout by default:
    # [[TP, FP],
    #  [FN, TN]]
    costs = [[10.0, -4.0], [-7.0, 1.0]]
    expected = cm.get_expected_value(costs)

    # ((3*10) + (1*-4) + (2*-7) + (4*1)) / 10 = 1.6
    assert expected == pytest.approx(1.6)


def test_confusion_matrix_expected_value_normal_display():
    y_true = ['No', 'No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes']
    y_pred = ['No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes']
    cm = ConfusionMatrix(y_true, y_pred, labels=['No', 'Yes'])

    # Costs follow normal layout:
    # [[TN, FP],
    #  [FN, TP]]
    costs = [[1.0, -4.0], [-7.0, 10.0]]
    expected = cm.get_expected_value(costs, display='normal')

    # ((4*1) + (1*-4) + (2*-7) + (3*10)) / 10 = 1.6
    assert expected == pytest.approx(1.6)


def test_confusion_matrix_infers_labels_from_python_lists():
    y_true = ['cat', 'dog', 'cat', 'dog']
    y_pred = ['cat', 'cat', 'cat', 'dog']
    cm = ConfusionMatrix(y_true, y_pred)
    assert cm.matrix.shape == (2, 2)


def test_confusion_matrix_summary_prints_readable_report(capsys):
    y_true = ['No', 'No', 'No', 'Yes', 'Yes', 'Yes']
    y_pred = ['No', 'Yes', 'No', 'Yes', 'No', 'Yes']
    cm = ConfusionMatrix(y_true, y_pred, labels=['No', 'Yes'])

    returned = cm.summary()
    out = capsys.readouterr().out

    assert isinstance(returned, pd.DataFrame)
    assert "Confusion Matrix and Statistics" in out
    assert "Overall Statistics:" in out
    assert "Statistics by Class:" in out
    assert "Binary Counts (Positive =" in out


@pytest.fixture
def binary_data():
    np.random.seed(42)
    x1 = np.random.normal(0, 1, 80)
    x2 = np.random.normal(0, 1, 80)
    y = np.where(x1 + x2 > 0, 'A', 'B')
    return pd.DataFrame({'x1': x1, 'x2': x2, 'class': y})


@pytest.mark.parametrize(
    'factory',
    [
        lambda df: fit_knn_classifier(df, formula='class ~ x1 + x2', k=3),
        lambda df: fit_logit(df, formula='class ~ x1 + x2', max_iter=2000),
        lambda df: fit_nn(df, formula='class ~ x1 + x2', hidden_layers=[3], solver='lbfgs', max_iter=1000),
        lambda df: fit_svm(df, formula='class ~ x1 + x2', C=1.0),
        lambda df: fit_tree(df, formula='class ~ x1 + x2', max_depth=3),
        lambda df: fit_lda(df, formula='class ~ x1 + x2'),
    ],
)
def test_models_expose_get_confusion_matrix(binary_data, factory):
    model = factory(binary_data)
    cm = model.get_confusion_matrix()

    assert isinstance(cm, ConfusionMatrix)
    assert cm.matrix.shape == (2, 2)
    assert np.array_equal(cm.matrix, model.get_score(binary_data)['confusion_matrix'])


@pytest.mark.parametrize(
    'factory',
    [
        lambda df: fit_knn_classifier(df, formula='class ~ x1 + x2', k=3),
        lambda df: fit_logit(df, formula='class ~ x1 + x2', max_iter=2000),
        lambda df: fit_nn(df, formula='class ~ x1 + x2', hidden_layers=[3], solver='lbfgs', max_iter=1000),
        lambda df: fit_svm(df, formula='class ~ x1 + x2', C=1.0),
        lambda df: fit_tree(df, formula='class ~ x1 + x2', max_depth=3),
        lambda df: fit_lda(df, formula='class ~ x1 + x2'),
    ],
)
def test_display_parameter_flows_through_score_and_summary(binary_data, factory, monkeypatch):
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    model = factory(binary_data)

    score_report = model.score(binary_data, display='inverted')
    assert 'confusion_matrix' in score_report

    summary_result = model.summary(display='inverted')
    assert summary_result is None
