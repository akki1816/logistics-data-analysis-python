
import pandas as pd
import matplotlib.pyplot as plt

# Sample logistics data
data = {
    "Hub": ["Kolkata", "Howrah", "Durgapur", "Kolkata",
            "Howrah", "Durgapur", "Kolkata", "Howrah"],
    "Month": ["January", "January", "January", "February",
              "February", "February", "March", "March"],
    "Shipments": [500, 450, 400, 420, 380, 350, 350, 300],
    "Delayed": [40, 35, 30, 65, 60, 55, 90, 75]
}

df = pd.DataFrame(data)

# Calculate delay percentage
df["Delay_Percentage"] = (
    df["Delayed"] / df["Shipments"] * 100
).round(2)

print("KOLKATA LOGISTICS ANALYSIS")
print("--------------------------")
print(df)

# Hub-wise analysis
print("\nHub-wise delayed shipments:")
print(df.groupby("Hub")["Delayed"].sum())

# Overall performance
total = df["Shipments"].sum()
delayed = df["Delayed"].sum()

print("\nTotal Shipments:", total)
print("Total Delayed:", delayed)
print("Overall Delay Percentage:",
      round(delayed / total * 100, 2), "%")

# Identify highest delay hub
hub_delays = df.groupby("Hub")["Delayed"].sum()
print("\nHighest Delay Hub:", hub_delays.idxmax())

# Create chart
hub_delays.plot(kind="bar", title="Delayed Shipments by Hub")
plt.xlabel("Hub")
plt.ylabel("Delayed Shipments")
plt.tight_layout()
plt.show()
