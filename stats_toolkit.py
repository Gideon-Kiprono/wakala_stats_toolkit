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
    if not values:
        raise ValueError("Cannot calculate the mean of an empty list")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate median of an empty list")
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid_val = n // 2
    if n % 2 != 0:
        return sorted_vals[mid_val]
    return (sorted_vals[mid_val - 1] + sorted_vals[mid_val]) / 2.0



def mode(values: list[float]) -> list[float]:
    """Return every value that occurs most often (a list, because there can be ties or none)."""
    if not values:
        raise ValueError("Cannot calculate mode of an empty list")
    
    frequency = {}
    for num in values:
        frequency[num] = frequency.get(num, 0) + 1

    max_freq = max(frequency.values())

    if max_freq == 1:
        return []
    

    return sorted(key for key, count in frequency.items() if count == max_freq)


    


def variance(values: list[float], sample: bool = True) -> float:
    """Sample variance (n - 1) by default; population variance (n) when sample is False."""

    n = len(values)

    if n == 0:
        raise ValueError("Variance requires at least one value.")
    if sample and n == 1:
        raise ValueError("Sample variance requires at least two values (degrees of freedom n-1).")


    mean_value = mean(values)
    squared_diff_sum = sum((x - mean_value) ** 2 for x in values) 
    denominator = (n - 1) if sample else n
    return squared_diff_sum / denominator

def std_dev(values: list[float], sample: bool = True) -> float:
    return math.sqrt(variance(values, sample=sample))


def _quantile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation quantile (same as NumPy's default, 'linear')."""
    n = len(sorted_vals)
    pos = p * (n - 1)          # fractional position in the sorted list
    i = int(pos)               # index just below that position
    f = pos - i                # how far past index i we are (0 to <1)
    if i + 1 >= n:             # p = 1.0 or single value: nothing above to blend with
        return sorted_vals[i]
    return sorted_vals[i] + f * (sorted_vals[i + 1] - sorted_vals[i])

def quartiles(values: list[float]) -> tuple[float, float, float]:
    """Return (Q1, Q2, Q3). Document which quartile method you chose."""
    if not values:
        raise ValueError("Cannot calculate quartiles of an empty list")
    s = sorted(values)
    return _quantile(s, 0.25), _quantile(s, 0.50),_quantile(s, 0.75)


def iqr_outliers(values: list[float], k: float = 1.5) -> dict:
    """Return the IQR, the lower and upper fences, and the values outside them."""
    q1, _, q3 = quartiles(values)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k *iqr
    outliers = [v for v in values if v < lower or v > upper]
    return {"iqr": iqr, "lower": lower, "upper": upper, "outliers": outliers}


class StatisticsAnalyzer:
    """Wraps one numeric column and exposes every statistic above as a method."""

    def __init__(self, values: list[float], name: str = "values",
                rejected: list[tuple[int, str]] | None = None,
                labels: list[tuple[str, str]] | None = None) -> None:

        if labels is not None and len(labels) != len(values):
            raise ValueError("labels must have one (date, agent_id) pair per value")
        self.values =list(values)
        self.name = name
        self.rejected = list(rejected) if rejected else []
        self.labels = list(labels) if labels else None


    @classmethod
    def from_csv(cls, path: Path, column: str) -> "StatisticsAnalyzer":
        values, rejected = load_column(path, column)
        bad_rows = {row for row, _ in rejected}
        labels = []
        with open(path, mode="r", newline="", encoding="utf-8") as file:
            for row_idx, row in enumerate(csv.DictReader(file), start=2):
                if row_idx not in bad_rows:
                    labels.append((row.get("date", ""), row.get("agent_id", "")))
        return cls(values, name=column, rejected=rejected, labels=labels)


    def summary(self) -> dict:
        v = self.values
        q1, _, q3 = quartiles(v)
        fences = iqr_outliers(v)
        return {
            "count": len(v), "mean": mean(v), "median": median(v), "mode": mode(v),
            "variance": variance(v), "std_dev": std_dev(v),
            "min": min(v), "max": max(v),
            "q1": q1, "q3": q3, "iqr": fences["iqr"],
            "lower": fences["lower"], "upper": fences["upper"],
            "outliers": fences["outliers"],
        }

    def outlier_details(self) -> list[tuple]:
        """(value, date, agent_id) for each outlier; date/agent are '' without labels."""
        fences = iqr_outliers(self.values)
        lo, hi = fences["lower"], fences["upper"]
        details = []
        for i, v in enumerate(self.values):
            if v < lo or v > hi:
                date, agent = self.labels[i] if self.labels else ("", "")
                details.append((v, date, agent))
        return details
    

    def without_outliers(self) -> "StatisticsAnalyzer":
        fences = iqr_outliers(self.values)
        lo, hi = fences["lower"], fences["upper"]
        keep = [i for i, v in enumerate(self.values) if lo <= v <= hi]
        return StatisticsAnalyzer(
            [self.values[i] for i in keep],
            name=self.name,
            rejected=self.rejected,
            labels=[self.labels[i] for i in keep] if self.labels else None,
        )

    @staticmethod
    def _reason(raw: str) -> str:
        """Why a raw value was rejected."""
        text = raw.strip()
        if text == "":
            return "missing"
        try:
            float(text)
            return "not finite"      # nan / inf parse as floats but are unusable
        except ValueError:
            return "not a number"

    @staticmethod
    def _fmt(x: float) -> str:
        return f"{x:,.2f}"

    def report(self) -> str:
        clean = self.without_outliers()
        a, b = self.summary(), clean.summary()
        fmt = self._fmt

        def mode_text(s: dict) -> str:
            return ", ".join(fmt(m) for m in s["mode"]) if s["mode"] else "none"

        def row(label: str, left: str, right: str) -> str:
            return f"{label:<20}{left:<26}{right}"

        lines = [
            f"=== Statistics report: {self.name} ===",
            f"Rows read         : {len(self.values) + len(self.rejected)}",
            f"Valid values      : {len(self.values)}",
            f"Rejected values   : {len(self.rejected)}",
        ]
        for row_no, raw in self.rejected:
            lines.append(f"  row {row_no}: '{raw}' ({self._reason(raw)})")
        lines += [
            "",
            row("", "all data", "outliers removed"),
            row("count", str(a["count"]), str(b["count"])),
            row("mean", fmt(a["mean"]), fmt(b["mean"])),
            row("median", fmt(a["median"]), fmt(b["median"])),
            row("mode", mode_text(a), mode_text(b)),
            row("variance (sample)", fmt(a["variance"]), fmt(b["variance"])),
            row("std dev (sample)", fmt(a["std_dev"]), fmt(b["std_dev"])),
            row("min / max", f"{fmt(a['min'])} / {fmt(a['max'])}",
                f"{fmt(b['min'])} / {fmt(b['max'])}"),
            row("Q1 / Q3", f"{fmt(a['q1'])} / {fmt(a['q3'])}",
                f"{fmt(b['q1'])} / {fmt(b['q3'])}"),
            row("IQR", fmt(a["iqr"]), fmt(b["iqr"])),
            "",
            f"Outlier fences      : lower {fmt(a['lower'])}, upper {fmt(a['upper'])}",
        ]
        details = self.outlier_details()
        if details:
            parts = [f"{fmt(v)} ({d}, {ag})" for v, d, ag in details]
            lines.append("Outliers            : " + ", ".join(parts))
        else:
            lines.append("Outliers            : none")
        return "\n".join(lines)


def compare_with_numpy(analyzer: StatisticsAnalyzer) -> dict[str, bool]:
    """Import numpy HERE, recompute each statistic and return {name: matches}."""
    import numpy as np

    v = np.array(analyzer.values, dtype=float)
    s = analyzer.summary()
    q1, q3 = np.percentile(v, [25, 75])      # default method is 'linear'
    return {
        "mean": math.isclose(s["mean"], float(np.mean(v))),
        "median": math.isclose(s["median"], float(np.median(v))),
        "variance": math.isclose(s["variance"], float(np.var(v, ddof=1))),
        "std_dev": math.isclose(s["std_dev"], float(np.std(v, ddof=1))),
        "quartiles": math.isclose(s["q1"], float(q1)) and math.isclose(s["q3"], float(q3)),
    }

def run_unit_tests() -> str:
    """Run the test suite quietly and return the 'Ran n tests ... OK' lines."""
    import io
    import time
    import unittest

    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*.py")
    start = time.perf_counter()
    result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    elapsed = time.perf_counter() - start
    status = "OK" if result.wasSuccessful() else f"FAILED ({len(result.failures)} failures, {len(result.errors)} errors)"
    return f"Ran {result.testsRun} tests in {elapsed:.3f}s\n{status}"


if __name__ == "__main__":
    analyzer = StatisticsAnalyzer.from_csv(DATA_PATH, "daily_deposits_kes")
    print(analyzer.report())

    checks = compare_with_numpy(analyzer)
    flag = lambda ok: "MATCH" if ok else "MISMATCH"


    output = "\n".join([
        analyzer.report(),
        "",
        "=== NumPy verification ===",
        f"mean      {flag(checks['mean'])}",
        f"median    {flag(checks['median'])}",
        f"variance  {flag(checks['variance'])}  (ddof=1)",
        f"std dev   {flag(checks['std_dev'])}",
        f"Q1 / Q3   {flag(checks['quartiles'])}  (method=linear)",
        "",
        "=== Unit tests ===",
        run_unit_tests(),
    ])

    print(output)

    REPORT_PATH = Path("output") / "report.txt"
    REPORT_PATH.write_text(output + "\n", encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH.resolve()}")


    


