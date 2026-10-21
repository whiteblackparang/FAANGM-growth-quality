import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faangm_quality_score import (
    build_company,
    candidate_facts,
    latest_common_quarter,
    percentile,
    quarterly_series,
    score_companies,
    ttm,
)

QUARTERS = [
    ("2024-01-01", "2024-03-31"), ("2024-04-01", "2024-06-30"),
    ("2024-07-01", "2024-09-30"), ("2024-10-01", "2024-12-31"),
    ("2025-01-01", "2025-03-31"), ("2025-04-01", "2025-06-30"),
    ("2025-07-01", "2025-09-30"), ("2025-10-01", "2025-12-31"),
]


def quarter_rows(values):
    return [{"start": s, "end": e, "val": v, "form": "10-Q", "filed": e}
            for (s, e), v in zip(QUARTERS, values)]


def series_of(values, ends):
    return pd.Series(values, index=pd.to_datetime(ends), dtype=float)


def test_quarterly_series_derives_second_quarter_from_six_month_value():
    rows = [
        {"start": "2025-01-01", "end": "2025-03-31", "val": 100, "form": "10-Q", "filed": "2025-05-01"},
        {"start": "2025-01-01", "end": "2025-06-30", "val": 230, "form": "10-Q", "filed": "2025-08-01"},
    ]

    result = quarterly_series(rows)

    assert result.loc[pd.Timestamp("2025-03-31")] == 100
    assert result.loc[pd.Timestamp("2025-06-30")] == 130


def test_quarterly_series_derives_fourth_quarter_from_annual_value():
    rows = [
        {"start": "2025-01-01", "end": "2025-03-31", "val": 100, "form": "10-Q", "filed": "2025-05-01"},
        {"start": "2025-04-01", "end": "2025-06-30", "val": 110, "form": "10-Q", "filed": "2025-08-01"},
        {"start": "2025-07-01", "end": "2025-09-30", "val": 120, "form": "10-Q", "filed": "2025-11-01"},
        {"start": "2025-01-01", "end": "2025-12-31", "val": 460, "form": "10-K", "filed": "2026-02-01"},
    ]

    result = quarterly_series(rows)

    assert result.loc[pd.Timestamp("2025-12-31")] == 130


def test_ttm_sums_last_four_quarters():
    series = series_of([10, 20, 30, 40, 50],
                       ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31"])

    assert ttm(series, pd.Timestamp("2025-03-31")) == 140


def test_ttm_is_nan_when_a_quarter_is_missing():
    series = series_of([10, 20, 40, 50],
                       ["2024-03-31", "2024-06-30", "2024-12-31", "2025-03-31"])

    assert np.isnan(ttm(series, pd.Timestamp("2025-03-31")))


def test_candidate_facts_fills_missing_period_from_older_tag():
    newer = {"units": {"USD": [{"start": "2025-04-01", "end": "2025-06-30", "val": 50}]}}
    older = {"units": {"USD": [
        {"start": "2025-01-01", "end": "2025-03-31", "val": 40},
        {"start": "2025-04-01", "end": "2025-06-30", "val": 999},
    ]}}
    facts = {"facts": {"us-gaap": {"Revenues": newer, "SalesRevenueNet": older}}}

    rows = candidate_facts(facts, ["Revenues", "SalesRevenueNet"])

    assert sorted(r["val"] for r in rows) == [40, 50]


def test_latest_common_quarter_needs_four_quarters_for_ttm():
    ends = ["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31"]
    series = {
        "revenue": series_of([1] * 6, ends),
        "operating_income": series_of([1] * 6, ends),
        "cfo": series_of([1] * 5, ends[:5]),
    }

    assert latest_common_quarter(series) == pd.Timestamp("2024-12-31")


def test_build_company_calculates_ttm_growth_and_fcf_margin():
    def facts_for(values):
        return {"units": {"USD": quarter_rows(values)}}

    facts = {"facts": {"us-gaap": {
        "Revenues": facts_for([100] * 4 + [120] * 4),
        "OperatingIncomeLoss": facts_for([20] * 8),
        "NetCashProvidedByUsedInOperatingActivities": facts_for([30] * 8),
        "PaymentsToAcquirePropertyPlantAndEquipment": facts_for([10] * 8),
        "ResearchAndDevelopmentExpense": facts_for([5] * 8),
    }}}

    row, _ = build_company("Test", {"ticker": "TST"}, facts)

    assert row["revenue_growth_yoy"] == pytest.approx(0.20)
    assert row["operating_margin"] == pytest.approx(80 / 480)
    assert row["fcf_margin"] == pytest.approx(80 / 480)


def test_percentile_returns_values_between_zero_and_one_hundred():
    result = percentile(pd.Series([10, 20, 30]))

    assert result.min() >= 0
    assert result.max() <= 100
    assert result.iloc[0] < result.iloc[-1]


def test_quality_score_ranks_higher_quality_company_first():
    data = pd.DataFrame({
        "company": ["Company A", "Company B"],
        "revenue_growth_yoy": [0.30, 0.05],
        "operating_margin": [0.40, 0.10],
        "fcf_margin": [0.25, 0.02],
        "investment_efficiency": [1e-10, 1e-12],
    })
    weights = {"revenue_growth": 0.30, "operating_margin": 0.25,
               "fcf_margin": 0.25, "investment_efficiency": 0.20}

    result = score_companies(data, weights)

    assert result.iloc[0]["company"] == "Company A"
    assert result.iloc[0]["quality_score"] > result.iloc[1]["quality_score"]