import pandas as pd


# Load housing data from XML file
data = pd.read_xml("data/lejemaal.xml")

# Convert Danish column names to English
data.rename(
    columns={
        "LejemaalId": "id",
        "AfdId": "department_id",
        "SelId": "company_id",
        "Adresse": "address",
        "PostBy": "postal_code_and_city",
        "Lejemaalstype": "property_type",
        "Lejlighedstype": "apartment_type",
        "Rum": "rooms",
        "Areal": "area",
        "BBRAreal": "bbr_area",
        "Indskud": "deposit",
        "NettoHusleje": "net_rent",
        "BruttoHusleje": "gross_rent",
    },
    inplace=True,
)

# Fix the numeric columns to handle different number formats
numeric_columns = ["area", "bbr_area", "deposit", "net_rent", "gross_rent"]


def parse_numeric_value(value):
    """
    Parse numbers from the XML data, handling both regular decimals and scientific notation.
    Returns rounded values with 2 decimal places.
    """
    try:
        value = str(value).strip()
        parsed_value = float(value)
        return round(parsed_value, 2)
    except ValueError as e:
        print(f"Error parsing value: '{value}'")
        return None


# Clean up the data by removing invalid entries
columns_to_check = ["postal_code_and_city", "area", "bbr_area", "net_rent"]
print("Rows before removing NAs:", data.shape[0])
data = data.dropna(subset=columns_to_check)
print("Rows after removed NAs:", data.shape[0])

# Remove entries with zero values in important fields
print("Rows before removing zero values:", data.shape[0])
critical_columns = ["rooms", "area", "bbr_area", "deposit", "net_rent", "gross_rent"]
mask = data[critical_columns].gt(0).all(axis=1)
data = data[mask]
print("Rows after removed zero values:", data.shape[0])

# Add price per square meter calculations
data["net_price_per_sqm"] = (data["net_rent"] / data["area"]).round(2)
data["gross_price_per_sqm"] = (data["gross_rent"] / data["area"]).round(2)


# Look for unusually low prices per square meter
price_per_sqm_threshold = 40  # DKK
low_price_cases = data[data["net_price_per_sqm"] < price_per_sqm_threshold]
low_rent_cases = data[data["net_rent"] < 1000]

print("\nUnusually low price per square meter cases:")
print(
    f"Found {len(low_price_cases + low_rent_cases)} entries with price/m² below {price_per_sqm_threshold} or withrent below 1000 DKK"
)
print("\nExample cases:")
print(
    low_price_cases[
        [
            "net_price_per_sqm",
            "area",
            "net_rent",
            "address",
            "property_type",
            "apartment_type",
        ]
    ].head(5)
)

# Remove unrealistic rent prices
print("Rows before removing unrealistic rents:", data.shape[0])
rent_mask = (
    (data["net_rent"] >= 1000)
    & (data["gross_rent"] >= 1000)
    & (data["net_price_per_sqm"] >= 40)
)
data = data[rent_mask]
print("Rows after removing unrealistic rents:", data.shape[0])


# Split postal code and city into separate columns
data["postal_code"] = data["postal_code_and_city"].str.extract(r"(\d{4})")
data["city"] = data["postal_code_and_city"].str.extract(r"\d{4}\s+(.*)")
data = data.drop("postal_code_and_city", axis=1)

# Export cleaned data to CSV
data.to_csv("data/lejemaal.csv", index=False)

# Look for unusual entries (unrealistically tiny apartments or very expensive ones)
print("\nUnusual entries in the dataset:")
area_threshold = 10  # m²
rent_threshold = 20000  # DKK

suspicious_cases = data[
    (data["area"] < area_threshold) | (data["gross_rent"] > rent_threshold)
]

print(f"Found {len(suspicious_cases)} unusual entries")
print("\nExample cases:")
print(
    suspicious_cases[
        ["area", "gross_rent", "address", "property_type", "apartment_type"]
    ].head(10)
)
