import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the housing data
df = pd.read_csv("data/lejemaal.csv")

# Create folders for our outputs
os.makedirs("output/statistics", exist_ok=True)
os.makedirs("output/plots", exist_ok=True)

# These are the main numerical features we'll analyze
numerical_columns = [
    "rooms",
    "area",
    "deposit",
    "net_rent",
    "gross_rent",
    "net_price_per_sqm",
    "gross_price_per_sqm",
]

# Export basic stats for our numerical columns
numerical_stats = df[numerical_columns].describe()
numerical_stats.to_csv("output/statistics/numerical_statistics.csv")

# Location-based Analysis
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# Top: Distribution of Properties by City
city_counts = df["city"].value_counts()
axes[0].bar(city_counts.index, city_counts.values)
axes[0].set_title("Where are the Properties Located?")
axes[0].set_xlabel("City")
axes[0].set_ylabel("Number of Properties")
axes[0].tick_params(axis="x", rotation=45)

# Bottom: Price per m² by Postal Code
postal_codes = sorted(df["postal_code"].unique())
postal_prices = [
    df[df["postal_code"] == code]["net_price_per_sqm"] for code in postal_codes
]
axes[1].boxplot(postal_prices, tick_labels=postal_codes)
axes[1].set_title("How do Prices Vary by Area?")
axes[1].set_xlabel("Postal Code")
axes[1].set_ylabel("Net Price per m²")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("output/plots/location_analysis.png", dpi=300, bbox_inches="tight")
plt.close()

# Rent Statistics by Location
fig, ax = plt.subplots(figsize=(12, 6))

# Rent Statistics by City
city_stats = df.groupby("city")["gross_rent"].agg(["mean", "median", "std"]).round(2)
x = range(len(city_stats.index))
width = 0.25

ax.bar(
    [i - width for i in x], city_stats["mean"], width, label="Mean Rent", color="blue"
)
ax.bar(x, city_stats["median"], width, label="Median Rent", color="green")
ax.bar([i + width for i in x], city_stats["std"], width, label="Std Dev", color="red")
ax.set_title("How do Rents Compare Across Cities?")
ax.set_xticks(x)
ax.set_xticklabels(city_stats.index, rotation=45)
ax.legend()

plt.tight_layout()
plt.savefig("output/plots/rent_by_location.png", dpi=300, bbox_inches="tight")
plt.close()

# Create correlation matrix for numerical features
correlation_matrix = df[numerical_columns].corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(correlation_matrix, cmap="coolwarm", aspect="auto")
plt.colorbar(im)

# Add correlation values to the heatmap
for i in range(len(numerical_columns)):
    for j in range(len(numerical_columns)):
        text = ax.text(
            j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha="center", va="center"
        )

ax.set_xticks(range(len(numerical_columns)))
ax.set_yticks(range(len(numerical_columns)))
ax.set_xticklabels(numerical_columns, rotation=45)
ax.set_yticklabels(numerical_columns)
plt.title("Correlation Matrix of Numerical Variables")
plt.tight_layout()
plt.savefig("output/plots/correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot property type distributions
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: Types of apartments
apartment_counts = df["apartment_type"].value_counts()
axes[0].bar(apartment_counts.index, apartment_counts.values)
axes[0].set_title("Distribution of Apartment Types")
axes[0].set_xlabel("Apartment Type")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=45)

# Right: Room count distribution
room_counts = df["rooms"].value_counts().sort_index()
axes[1].bar(room_counts.index, room_counts.values)
axes[1].set_title("Distribution of Number of Rooms")
axes[1].set_xlabel("Number of Rooms")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("output/plots/descriptive_statistics.png", dpi=300, bbox_inches="tight")
plt.close()

# Rent distribution analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: Distribution of net rent (excluding outliers)
rent_data = df[df["net_rent"] <= df["net_rent"].quantile(0.95)]
axes[0].hist(rent_data["net_rent"], bins=30, edgecolor="black")
axes[0].set_title("How is Net Rent Distributed?")
axes[0].set_xlabel("Net Rent (DKK)")
axes[0].set_ylabel("Number of Properties")

# Right: How rent varies by number of rooms
room_numbers = sorted(df["rooms"].unique())
room_rents = [df[df["rooms"] == room]["net_rent"] for room in room_numbers]
axes[1].boxplot(room_rents, tick_labels=room_numbers)
axes[1].set_title("How Does Rent Change with Room Count?")
axes[1].set_xlabel("Number of Rooms")
axes[1].set_ylabel("Net Rent (DKK)")

plt.tight_layout()
plt.savefig("output/plots/rent_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# Area vs rent analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: Area vs gross rent
filtered_df = df[(df["area"] <= 200) & (df["gross_rent"] <= 15000)]
scatter = axes[0].scatter(
    filtered_df["area"],
    filtered_df["gross_rent"],
    c=filtered_df["rooms"],
    cmap="viridis",
    alpha=0.6,
)
axes[0].set_title("How Does Area Affect Gross Rent?")
axes[0].set_xlabel("Area (m²)")
axes[0].set_ylabel("Gross Rent (DKK)")
plt.colorbar(scatter, ax=axes[0], label="Number of Rooms")

# Right: Area vs net rent
scatter = axes[1].scatter(
    filtered_df["area"],
    filtered_df["net_rent"],
    c=filtered_df["rooms"],
    cmap="viridis",
    alpha=0.6,
)
axes[1].set_title("How Does Area Affect Net Rent?")
axes[1].set_xlabel("Area (m²)")
axes[1].set_ylabel("Net Rent (DKK)")
plt.colorbar(scatter, ax=axes[1], label="Number of Rooms")

plt.tight_layout()
plt.savefig("output/plots/area_vs_rent.png", dpi=300, bbox_inches="tight")
plt.close()

# Price per square meter distribution analysis
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate IQR bounds for extreme outlier removal
Q1 = df["net_price_per_sqm"].quantile(0.25)
Q3 = df["net_price_per_sqm"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3.0 * IQR
upper_bound = Q3 + 3.0 * IQR

# Filter only the most extreme outliers
price_per_sqm_data = df[
    (df["net_price_per_sqm"] >= lower_bound) & (df["net_price_per_sqm"] <= upper_bound)
]

# Create histogram
ax.hist(price_per_sqm_data["net_price_per_sqm"], bins=30, edgecolor="black")
ax.set_title(
    "Distribution of Net Price per Square Meter (Excluding Outliers More Extreme than 3 IQR)"
)
ax.set_xlabel("Net Price per m² (DKK)")
ax.set_ylabel("Number of Properties")

plt.tight_layout()
plt.savefig("output/plots/price_per_sqm_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
