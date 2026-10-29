Day 3 – Statistical Analysis Toolkit

A statistics toolkit written in plain Python, applied to daily agent deposit data (daily_deposits_kes) for a mobile-money agent network. It loads and cleans the data, computes descriptive statistics by hand, flags outliers with the IQR rule, and produces a report. NumPy is used only to verify the results.

Project structure
day3-stats-toolkit/
├── stats_toolkit.py          # functions + StatisticsAnalyzer class + report runner
├── test_stats_toolkit.py     # unit tests (unittest)
├── data/agent_deposits.csv   # input data (40 rows)
├── output/stats_report.txt   # generated report
├── requirements.txt          # numpy (verification only)
└── README.md
How to run

Requirements: Python 3.10+ (the code uses X | None type hints) and NumPy for the verification step.

bash
pip install -r requirements.txt

Run everything from the project root (the folder containing data/), because the data path is relative:

bash
# Generate the report (prints it and saves it to output/stats_report.txt)
python stats_toolkit.py

# Run the unit tests
python -m unittest -v

pytest also works (pytest test_stats_toolkit.py) but is not required.

Approach

All calculations are implemented from scratch with plain Python. The statistics module is not used, and NumPy is imported only inside compare_with_numpy().

Function	What it does
load_column	Reads one CSV column; returns the valid floats and the rejected (row_number, raw_value) pairs
mean, median, mode	Central tendency. mode returns a list (ties give several values, no repeated value gives [])
variance, std_dev	Spread, with a sample switch
quartiles, iqr_outliers	Q1/Q2/Q3, IQR, fences and outliers
StatisticsAnalyzer	Wraps one column; from_csv, summary, without_outliers, outlier_details, report
compare_with_numpy	Recomputes the statistics with NumPy and returns {name: matches}

StatisticsAnalyzer.from_csv also keeps the rejected rows and a (date, agent_id) label for every valid value. The report needs these to list rejected rows and to say which day and agent each outlier came from. without_outliers() returns a new analyzer and leaves the original unchanged.

Rejected rows are matched to values by row number, not by value, so two identical deposits can never be confused.

Choices

Sample vs population. The report uses sample variance and standard deviation (divide by n - 1). The 35 valid values are treated as a sample of the agents' ongoing activity, not the complete population. Population versions are available with sample=False. A single value has no sample variance, so variance([x]) raises ValueError.

Quartile method. Linear interpolation between closest ranks: for n sorted values, quantile p sits at position p * (n - 1) and is interpolated between its two neighbours. This is NumPy's default (method="linear"), so the results can be verified directly.

Outlier rule. Tukey's IQR rule with k = 1.5: a value is an outlier if it is below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR. Values exactly on a fence are kept. Outliers are reported, and a second set of statistics is shown with them removed (a single pass, not repeated).

What counts as invalid. A value is rejected if, after stripping whitespace, it is:

empty or missing ("")
not a number (-, n/a, 12,40O with a letter O)
not finite (nan, inf)

Rejected values are never repaired or guessed. For example, 12,40O is not converted to 1240. Each rejection is listed in the report with its row number, raw text and reason. Row numbers match the CSV file as seen in a spreadsheet (header = row 1).

Assumptions
The column to analyse is daily_deposits_kes, in Kenyan shillings, and all values share the same unit.
The first line of the CSV is a header, and the file is UTF-8.
Rejected rows are excluded from every statistic; they are not treated as zero.
A negative amount would still be a valid number, so it would be kept, although none occur in this data.
Outliers are flagged, not deleted from the source data. Whether they are errors or real activity cannot be decided from the numbers alone.
Mode is rarely meaningful for two-decimal currency amounts, so none is the expected result here.
Results on the provided data
40 rows read, 35 valid, 5 rejected (rows 7, 10, 14, 25, 29).
4 outliers flagged outside the fences 8,530.19 to 28,445.41: 96,500.00, 1,200.00, 88,400.00 and 29,060.99. Three of them come from agent AG-104 (Nakuru).
	All data	Outliers removed
count	35	31
mean	22,400.00	18,349.65
median	18,805.47	18,506.72
std dev (sample)	18,111.32	3,180.63

The outliers pull the mean up by about 4,000 and inflate the standard deviation roughly 5.7 times, while the median moves by only about 300. This is why the median and IQR are more reliable summaries for data like this. The value 29,060.99 is only about 600 above the upper fence, so it is a borderline case that depends on the k = 1.5 convention.

All five NumPy checks (mean, median, variance with ddof=1, standard deviation, Q1/Q3 with the linear method) report MATCH.

Tests

test_stats_toolkit.py contains 12 unittest tests covering mean, median (odd and even lengths), mode (ties, single mode, no mode), empty-input errors, sample vs population variance, the single-value case, quartiles, outlier detection, rejected-value loading (including a quoted "12,40O" value and a missing column), and the analyzer's without_outliers and label alignment.