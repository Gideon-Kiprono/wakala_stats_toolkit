"""
Day 3 - Statistical Analysis Toolkit
Squad Python Specialization - Data Science, AI & Analytics

RULE: implement every calculation yourself with plain Python.
NumPy may be imported ONLY in compare_with_numpy(), to verify your results.
Do not use the statistics module.
"""
import csv
import math
from pathlib import Path

DATA_PATH = Path("data") / "agent_deposits.csv"


def load_column(path: Path, column: str) -> tuple[list[float], list[tuple[int, str]]]:
    """Return (valid_values, rejected) where rejected holds (row_number, raw_value) pairs."""
    
    valid_values = []
    rejected = []
    
    with open(path, mode='r', newline="", encoding="utf-8") as file:
        reader= csv.DictReader(file)
        

        if reader.fieldnames is None or column not in reader.fieldnames:
            raise KeyError(f"Column {column} not found in CSV header")

        for row_idx, row in enumerate(reader, start= 2):
            raw_value = row[column]

            if raw_value is None:
                rejected.append((row_idx, ""))
                continue

            stripped_value = raw_value.strip()

            try:
                parsed_value = float(stripped_value)

                if math.isnan(parsed_value) or math.isinf(parsed_value):
                    rejected.append((row_idx,stripped_value))

                else:
                    valid_values.append(parsed_value)

            except ValueError:
                rejected.append((row_idx,raw_value ))

        return valid_values, rejected




def mean(values: list[float]) -> float:
    raise NotImplementedError


def median(values: list[float]) -> float:
    raise NotImplementedError


def mode(values: list[float]) -> list[float]:
    """Return every value that occurs most often (a list, because there can be ties or none)."""
    raise NotImplementedError


def variance(values: list[float], sample: bool = True) -> float:
    """Sample variance (n - 1) by default; population variance (n) when sample is False."""
    raise NotImplementedError


def std_dev(values: list[float], sample: bool = True) -> float:
    raise NotImplementedError


def quartiles(values: list[float]) -> tuple[float, float, float]:
    """Return (Q1, Q2, Q3). Document which quartile method you chose."""
    raise NotImplementedError


def iqr_outliers(values: list[float], k: float = 1.5) -> dict:
    """Return the IQR, the lower and upper fences, and the values outside them."""
    raise NotImplementedError


class StatisticsAnalyzer:
    """Wraps one numeric column and exposes every statistic above as a method."""

    def __init__(self, values: list[float], name: str = "values") -> None:
        raise NotImplementedError

    @classmethod
    def from_csv(cls, path: Path, column: str) -> "StatisticsAnalyzer":
        raise NotImplementedError

    def summary(self) -> dict:
        raise NotImplementedError

    def without_outliers(self) -> "StatisticsAnalyzer":
        raise NotImplementedError

    def report(self) -> str:
        raise NotImplementedError


def compare_with_numpy(analyzer: StatisticsAnalyzer) -> dict[str, bool]:
    """Import numpy HERE, recompute each statistic and return {name: matches}."""
    raise NotImplementedError


if __name__ == "__main__":

    cleaned,skipped = load_column(DATA_PATH, "daily_deposits_kes")

    print(f"{len(cleaned)} valid, {len(skipped)} skipped")
    print("First rejects:" ,skipped[:5])
