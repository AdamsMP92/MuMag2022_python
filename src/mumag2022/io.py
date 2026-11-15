"""Input and output helpers independent of any GUI framework."""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .data import FieldSeries, ScatteringCurve


def parse_legacy_filename(filename: str) -> tuple[float, float, float]:
    """Return applied field, saturation magnetization, and demag field in mT."""
    fields = Path(filename).stem.split("_")
    if len(fields) != 4:
        raise ValueError(f"{filename!r} does not follow index_H0_Ms_Hd.ext")
    try:
        return tuple(float(value) for value in fields[1:])
    except ValueError as error:
        raise ValueError(f"could not parse magnetic metadata from {filename!r}") from error


def load_legacy_payloads(files: Mapping[str, bytes]) -> FieldSeries:
    """Load uploaded legacy files from an in-memory filename/payload mapping."""
    curves = []
    for filename, payload in files.items():
        if not payload:
            raise ValueError(f"{filename!r} is empty")
        applied, magnetization, demag = parse_legacy_filename(filename)
        text = payload.decode("utf-8-sig")
        first_line = next((line for line in text.splitlines() if line.strip()), "")
        delimiter = "," if "," in first_line else None
        values = np.loadtxt(io.StringIO(text), delimiter=delimiter)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"{filename!r} must contain exactly three columns")
        curves.append(
            ScatteringCurve(
                values[:, 0],
                values[:, 1],
                values[:, 2],
                applied,
                magnetization,
                demag,
                label=filename,
            )
        )
    return FieldSeries(tuple(curves))


def load_legacy_directory(directory: str | Path) -> FieldSeries:
    """Load all CSV/DAT/TXT legacy curves in one directory."""
    directory = Path(directory)
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in {".csv", ".dat", ".txt"}
    )
    if not paths:
        raise ValueError(f"no supported data files found in {directory}")
    return load_legacy_payloads({path.name: path.read_bytes() for path in paths})


def rows_to_csv(headers: Sequence[str], rows) -> str:
    """Serialize simple numerical rows without adding a dataframe dependency."""
    buffer = io.StringIO()
    buffer.write(",".join(headers) + "\n")
    for row in rows:
        buffer.write(",".join(_format(value) for value in row) + "\n")
    return buffer.getvalue()


def _format(value) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    return str(value)
