
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# KOLKATA LOGISTICS ROOT CAUSE ANALYSIS
# Synthetic demonstration data - not real company records

np.random.seed(42)

OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

N = 150000

print("Starting Logistics Root Cause Analysis...")

# 1. Generate shipment data
hubs = [f"H{i:02d}" for i in range(1, 13)]

shipments = pd.DataFrame({
    "shipment_id": range(N),
    "hub_id": np.random.choice(hubs, N),
    "service_type": np.random.choice(
        ["Standard", "Express", "Economy"], N
    ),
    "customer_segment": np.random.choice(
        ["Retail", "SME", "Enterprise"], N
    ),
    "quarter": np.random.choice(["Q1", "Q2"], N),
    "distance_km": np.random.randint(20, 650, N)
})

# 2. Simulate operational factors
q2 = (shipments["quarter"] == "Q2").astype(int)

shipments["route_delay"] = np.maximum(
    0, np.random.normal(1 + 0.6*q2, 0.8, N)
)

shipments["breakdown"] = np.random.binomial(
    1, 0.03 + 0.025*q2, N
)

shipments["warehouse_delay"] = np.random.gamma(
    2, 1.2, N
) + 0.6*q2

shipments["festive_week"] = np.random.binomial(
    1, 0.12, N
)

# 3. Calculate delivery time
shipments["delivery_hours"] = (
    5
    + shipments["distance_km"]/45
    + shipments["route_delay"]
    + shipments["breakdown"]*4
    + shipments["warehouse_delay"]
    + np.random.normal(0, 2, N)
)

shipments["promised_hours"] = (
    12 + shipments["distance_km"]/45
)

shipments["on_time"] = (
    shipments["delivery_hours"]
    <= shipments["promised_hours"]
).astype(int)

# 4. Create hub aliases
hub_mapping = {
    "OLD_H01": "H01",
    "OLD_H03": "H03",
    "KOL-07": "H07",
    "EAST-12": "H12"
}

shipments["hub_code"] = shipments["hub_id"]

for old, new in hub_mapping.items():
    shipments.loc[
        shipments["hub_id"] == new, "hub_code"
    ] = old

shipments["hub_id"] = shipments["hub_code"].replace(
    hub_mapping
)

# 5. Data quality checks
shipments = shipments.drop_duplicates(
    subset=["shipment_id"]
)

shipments = shipments.dropna()

print("\nData Cleaning Completed")
print("Total Shipments:", len(shipments))
print("Total Hubs:", shipments["hub_id"].nunique())

# 6. KPI analysis
kpi = shipments.groupby("quarter").agg(
    total_shipments=("shipment_id", "count"),
    on_time_rate=("on_time", "mean"),
    average_delivery_hours=("delivery_hours", "mean"),
    average_route_delay=("route_delay", "mean"),
    breakdown_rate=("breakdown", "mean"),
    warehouse_delay=("warehouse_delay", "mean")
)

kpi["on_time_percentage"] = (
    kpi["on_time_rate"] * 100
)

print("\nQUARTERLY KPI REPORT")
print(kpi)

kpi.to_csv(OUTPUT / "quarterly_kpis.csv")

# 7. Hypothesis testing

results = []

# H1: Route delays differ between quarters
a = shipments.loc[
    shipments.quarter == "Q1", "route_delay"
]
b = shipments.loc[
    shipments.quarter == "Q2", "route_delay"
]

t, p = stats.ttest_ind(a, b, equal_var=False)

results.append(["H1 Route delays", t, p])

# H2: Vehicle breakdown and delivery performance
table = pd.crosstab(
    shipments["breakdown"], shipments["on_time"]
)

chi, p, _, _ = stats.chi2_contingency(table)

results.append(["H2 Vehicle breakdown", chi, p])

# H3: Warehouse delays differ between quarters
a = shipments.loc[
    shipments.quarter == "Q1", "warehouse_delay"
]
b = shipments.loc[
    shipments.quarter == "Q2", "warehouse_delay"
]

t, p = stats.ttest_ind(a, b, equal_var=False)

results.append(["H3 Warehouse delays", t, p])

# H4: Hub performance differences
table = pd.crosstab(
    shipments["hub_id"], shipments["on_time"]
)

chi, p, _, _ = stats.chi2_contingency(table)

results.append(["H4 Hub performance", chi, p])

# H5: Service type differences
table = pd.crosstab(
    shipments["service_type"], shipments["on_time"]
)

chi, p, _, _ = stats.chi2_contingency(table)

results.append(["H5 Service types", chi, p])

# H6: Festive demand differences
table = pd.crosstab(
    shipments["festive_week"], shipments["on_time"]
)

chi, p, _, _ = stats.chi2_contingency(table)

results.append(["H6 Festive season", chi, p])

hypotheses = pd.DataFrame(
    results,
    columns=["Hypothesis", "Test Statistic", "P Value"]
)

hypotheses["Decision"] = np.where(
    hypotheses["P Value"] < 0.05,
    "Reject Null Hypothesis",
    "Fail to Reject Null Hypothesis"
)

print("\nHYPOTHESIS TEST RESULTS")
print(hypotheses)

hypotheses.to_csv(
    OUTPUT / "hypothesis_log.csv", index=False
)

# 8. Hub-wise analysis
hub_report = shipments.groupby("hub_id").agg(
    shipments=("shipment_id", "count"),
    on_time_rate=("on_time", "mean"),
    average_delivery=("delivery_hours", "mean")
)

hub_report.to_csv(OUTPUT / "hub_analysis.csv")

# 9. Driver metrics
driver_metrics = shipments.groupby("quarter").agg(
    route_delay=("route_delay", "mean"),
    breakdown_rate=("breakdown", "mean"),
    warehouse_delay=("warehouse_delay", "mean")
)

driver_metrics.to_csv(
    OUTPUT / "driver_metrics.csv"
)

# 10. Weekly early-warning demonstration
shipments["week"] = np.arange(N) % 26 + 1

weekly = shipments.groupby("week").agg(
    shipments=("shipment_id", "count"),
    on_time_rate=("on_time", "mean"),
    warehouse_delay=("warehouse_delay", "mean")
)

weekly["previous_average"] = (
    weekly["on_time_rate"]
    .shift(1)
    .rolling(4, min_periods=4)
    .mean()
)

weekly["warning"] = (
    weekly["on_time_rate"]
    < weekly["previous_average"] - 0.03
)

weekly.to_csv(
    OUTPUT / "weekly_early_warning.csv"
)

# 11. Create delivery performance chart
kpi["on_time_percentage"].plot(
    kind="bar",
    title="Quarterly On-Time Delivery Performance"
)

plt.ylabel("On-Time Delivery (%)")
plt.xlabel("Quarter")
plt.tight_layout()

plt.savefig(
    OUTPUT / "delivery_performance.png"
)

plt.close()

# 12. Save cleaned data sample
shipments.head(1000).to_csv(
    OUTPUT / "cleaned_data_sample.csv",
    index=False
)

print("\nROOT CAUSE ANALYSIS COMPLETED!")
print("Generated files are available in outputs/")
print("NOTE: All findings use synthetic demonstration data.")
