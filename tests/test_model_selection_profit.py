import numpy as np
import pandas as pd

from cuanalytics.model_selection import calculate_curve, plot_profit


class DummyBinaryModel:
    def __init__(self, scores, target="y"):
        self._scores = np.asarray(scores, dtype=float)
        self.target = target
        self.classes = [0, 1]

    def predict_proba(self, df):
        if len(df) != len(self._scores):
            raise ValueError("Length mismatch")
        return np.column_stack([1.0 - self._scores, self._scores])


def test_profit_curve_is_ranked_and_uses_population_fraction():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 0, 1, 0]})
    model = DummyBinaryModel(scores=[0.9, 0.8, 0.7, 0.1])
    result = calculate_curve(models=[model], test_df=df, positive_class=1, model_names=["M"])
    profit = result.get_profit()["M"]
    data = profit["data"]

    assert "population_frac" in data.columns
    assert data["population_frac"].iloc[0] == 0.0
    assert data["threshold"].iloc[0] == 1.0
    assert data["population_frac"].iloc[-1] == 1.0
    assert np.all(np.diff(data["population_frac"].to_numpy()) >= 0)
    assert np.all(np.diff(data["threshold"].to_numpy()) <= 0)


def test_profit_curve_matches_cumulative_targeting_profit():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 0, 1, 0]})
    model = DummyBinaryModel(scores=[0.9, 0.8, 0.7, 0.1])
    result = calculate_curve(
        models=[model],
        test_df=df,
        positive_class=1,
        model_names=["M"],
        profit_config={
            "tp_value": 10.0,
            "fp_value": -1.0,
            "tn_value": 0.0,
            "fn_value": 0.0,
            "fixed_value": 0.0,
        },
    )
    profit = result.get_profit()["M"]
    data = profit["data"]

    expected_pop = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    expected_profit = np.array([0.0, 10.0, 9.0, 19.0, 18.0])
    assert np.allclose(data["population_frac"].to_numpy(), expected_pop)
    assert np.allclose(data["profit"].to_numpy(), expected_profit)
    assert profit["max_profit"] == 19.0
    assert np.isclose(profit["max_profit_threshold"], 0.7)
    assert np.isclose(profit["max_profit_population_frac"], 0.75)


def test_plot_profit_uses_targeted_percentage_axis():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 0, 1, 0]})
    model = DummyBinaryModel(scores=[0.9, 0.8, 0.7, 0.1])
    result = calculate_curve(models=[model], test_df=df, positive_class=1, model_names=["M"])
    plotted = plot_profit(result, show=False)
    ax = plotted["ax"]
    data = plotted["data"]["M"]["data"]

    assert ax.get_xlabel() == "Test Instances Targeted (%)"
    line = ax.lines[0]
    assert np.allclose(line.get_xdata(), data["population_frac"].to_numpy() * 100)

