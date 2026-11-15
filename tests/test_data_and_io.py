import numpy as np
import pytest

from mumag2022 import FieldSeries, ScatteringCurve, load_legacy_payloads, parse_legacy_filename


def curve(field=100.0, q=None):
    if q is None:
        q = np.array([0.05, 0.1, 0.2])
    return ScatteringCurve(q, [3.0, 2.0, 1.0], [0.2, 0.2, 0.2], field, 1200.0, 10.0)


def test_curve_rejects_nonpositive_internal_field():
    with pytest.raises(ValueError, match="internal field"):
        ScatteringCurve([0.1, 0.2], [1.0, 2.0], [0.1, 0.1], 10.0, 1000.0, 10.0)


def test_series_never_silently_resamples_q_grid():
    with pytest.raises(ValueError, match="same q grid"):
        FieldSeries((curve(100.0), curve(200.0, np.array([0.05, 0.11, 0.2]))))


def test_legacy_payload_loader_parses_metadata_and_columns():
    payloads = {
        "1_100_1200_10.csv": b"0.05,3,0.2\n0.1,2,0.2\n0.2,1,0.2\n",
        "2_200_1200_10.csv": b"0.05,2,0.2\n0.1,1.5,0.2\n0.2,0.8,0.2\n",
    }
    series = load_legacy_payloads(payloads)
    assert series.intensity.shape == (2, 3)
    np.testing.assert_allclose(series.applied_fields_mT, [100.0, 200.0])
    assert parse_legacy_filename("3_61_1640_23.748285.csv") == (
        61.0,
        1640.0,
        23.748285,
    )
