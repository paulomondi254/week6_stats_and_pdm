# Week 6: Statistical Validation & Predictive Maintenance

A three-part analysis of synthetic pump sensor data: validating a maintenance hypothesis with statistics, engineering rolling features to capture degradation trends, and training a classifier to flag failing pumps before they break.

## Dataset

`pdm_synthetic_features.csv` — sensor readings for **24 pumps across 4 depots** (North, South, East, West), logged every 6 hours. 2,640 rows total. Each row includes:

- Raw signals: `vibration_g`, `temperature_c`, `pressure_psi`, `acoustic_db`, `motor_current_a`, `error_events`
- Context: `asset_id`, `depot`, `shift`, `asset_age_months`, `timestamp`
- Pre-built rolling features: `vibration_roll_mean_6`, `vibration_rms_6`, `vibration_peak_to_peak_6`, `temp_roll_mean_6`
- Labels: `health_status` (Healthy / Warning / Critical), `fault_type`, `time_to_failure_hours`, `failure_within_7_days`, `constant_sensor`

## Part 1: Statistical Validation

**Question:** Do pumps logged as `Critical` actually vibrate meaningfully more than `Healthy` pumps, or does that just look true?

- Independent-samples t-test (Welch's, unequal variance) comparing `vibration_g` between Healthy and Critical pumps
- Result: t = -56.35, p ≈ 6.7e-128 — the null hypothesis is rejected; the difference is real, not noise
- 95% confidence intervals: Healthy 1.876g (1.855–1.897), Critical 4.195g (4.116–4.273) — the intervals don't overlap, confirming the gap is both statistically significant and large enough to matter operationally
- A boxplot visualizes the separation between the two groups

## Part 2: PdM Feature Engineering

Raw sensor readings are noisy, so rolling (6-reading / 36-hour window) features are built per pump to capture trends rather than single spikes:

- `vibration_roll_mean_6` — rolling average vibration
- `vibration_rms_6` — rolling root-mean-square vibration
- `vibration_peak_to_peak_6` — rolling max-min spread
- `pressure_roll_std_6` — rolling pressure volatility

A case study on `PUMP-001` (which develops a fault partway through the year) shows all four features climbing together as the pump approaches failure — the rolling mean/RMS track the rising average level while peak-to-peak/pressure std track rising instability.

The binary target `is_failing` is derived from `health_status` (1 = Warning or Critical, 0 = Healthy). About 21% of readings are failing — an imbalanced but not extreme split.

## Part 3: Predictive Modeling

**Model:** Logistic regression predicting `is_failing` from the four engineered features plus raw `temperature_c` and `motor_current_a`, with an 75/25 stratified train/test split and standardized features.

**Results:**
- Accuracy: 95.5%
- Precision/Recall (Failing class): 0.95 / 0.83
- Confusion matrix: 6 false alarms (healthy flagged as failing), 24 missed failures (failing flagged as healthy)
- Most influential features (by standardized coefficient): `vibration_roll_mean_6`, `vibration_rms_6`, `temperature_c`, `motor_current_a`

**Business takeaway:** overall accuracy looks strong, but on this imbalanced target the confusion matrix matters more — missed failures (pump keeps running until it actually breaks) are the costly error compared to false alarms (an unnecessary but cheap maintenance check), so the model is a solid start but not yet good enough to fully trust for unplanned-downtime prevention.

## Files

- `week6_stats_and_pdm.ipynb` — full analysis notebook (all three parts above)
- `pdm_synthetic_features.csv` — input sensor dataset
- `Week6_Communication_Briefs_Paul.pdf` — accompanying communication brief
- `Capstone_Proposal_MasterMaisha.pdf` — related capstone proposal document
- `README.md` — this file

## Requirements

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
```

## Running It

1. Install dependencies: `pip install pandas numpy matplotlib seaborn scipy scikit-learn`
2. Open `week6_stats_and_pdm.ipynb` in Jupyter
3. Run all cells top to bottom
