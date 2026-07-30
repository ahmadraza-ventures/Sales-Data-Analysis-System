import pandas as pd
import numpy as np


# -------------------------------
# Load CSV File
# -------------------------------

df = pd.read_csv("sales_data.csv")

print("Original Dataset\n")
print(df)


# -------------------------------
# Dataset Information
# -------------------------------

print("\nDataset Information\n")
print(df.info())

# -------------------------------
# Missing Values
# -------------------------------

print("\nMissing Values\n")
print(df.isnull().sum())

# -------------------------------
# Remove Duplicate Rows
# -------------------------------

df = df.drop_duplicates()
print("\nDuplicate Rows Removed Successfully")


# -------------------------------
# Convert Price into Float
# -------------------------------

df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

# -------------------------------
# Fill Missing Price
# -------------------------------

df["price"] = df["price"].fillna(0)

# -------------------------------
# Remove Missing Quantity
# -------------------------------

df = df.dropna(subset=["quantity"])

# -------------------------------
# Remove Missing Date
# -------------------------------

df = df.dropna(subset=["date"])

# -------------------------------
# Convert Date
# -------------------------------

df["date"] = pd.to_datetime(df["date"])

# -------------------------------
# Clean Dataset
# -------------------------------

print("\nClean Dataset\n")

print(df)

print("\nData Cleaning Completed Successfully!")



# ==========================================
# PART 2 - NUMPY OPERATIONS
# ==========================================

print("\n========== NUMPY OPERATIONS ==========\n")

# -------------------------------
# Convert Columns to NumPy Arrays
# -------------------------------
price = df["price"].to_numpy()
quantity = df["quantity"].to_numpy()

print("Price Array:")
print(price)

print("\nQuantity Array:")
print(quantity)

# -------------------------------
# Calculate Revenue
# -------------------------------
revenue = price * quantity

# Add Revenue Column
df["revenue"] = revenue

print("\nRevenue for Each Order:")
print(df[["product", "price", "quantity", "revenue"]])

# -------------------------------
# Total Revenue
# -------------------------------
total_revenue = np.sum(revenue)

print("\nTotal Revenue:", total_revenue)

# -------------------------------
# Mean Price
# -------------------------------
mean_price = np.mean(price)

print("Mean Price:", mean_price)

# -------------------------------
# Median Price
# -------------------------------
median_price = np.median(price)

print("Median Price:", median_price)

# -------------------------------
# Standard Deviation
# -------------------------------
std_price = np.std(price)

print("Standard Deviation:", std_price)

print("\nNumPy Operations Completed Successfully!")



# ==========================================
# PART 3 - PANDAS ANALYSIS
# ==========================================

print("\n========== PANDAS ANALYSIS ==========\n")

# -------------------------------
# Total Revenue by Category
# -------------------------------
print("----- Revenue by Category -----")

category_revenue = df.groupby("category")["revenue"].sum()

print(category_revenue)


# -------------------------------
# Total Revenue by Region
# -------------------------------
print("\n----- Revenue by Region -----")

region_revenue = df.groupby("region")["revenue"].sum()

print(region_revenue)


# -------------------------------
# Top 5 Best Selling Products
# -------------------------------
print("\n----- Top 5 Best Selling Products -----")

top_products = (
    df.groupby("product")["quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

print(top_products)


# -------------------------------
# Month-wise Revenue Trend
# -------------------------------
print("\n----- Month-wise Revenue -----")

df["month"] = df["date"].dt.month_name()

monthly_revenue = df.groupby("month")["revenue"].sum()

print(monthly_revenue)

print("\nPandas Analysis Completed Successfully!")



# ==========================================
# PART 4 - OUTLIER DETECTION & SUMMARY REPORT
# ==========================================

print("\n========== OUTLIER DETECTION ==========\n")

# -------------------------------
# Detect Outliers using IQR Method
# -------------------------------
def detect_outliers(data):

    Q1 = data["price"].quantile(0.25)
    Q3 = data["price"].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - (1.5 * IQR)
    upper_limit = Q3 + (1.5 * IQR)

    outliers = data[
        (data["price"] < lower_limit) |
        (data["price"] > upper_limit)
    ]

    return outliers


outlier_data = detect_outliers(df)

print("Outlier Records:\n")
print(outlier_data)

# ==========================================
# SUMMARY REPORT
# ==========================================

print("\n========== SUMMARY REPORT ==========\n")

summary = df.groupby("category").agg({

    "revenue": "sum",
    "price": "mean",
    "order_id": "count"

})

summary.columns = [

    "Total Revenue",
    "Average Order Value",
    "Number of Orders"

]

print(summary)

# ==========================================
# EXPORT REPORTS
# ==========================================

summary.to_csv("summary_report.csv")

summary.to_excel("summary_report.xlsx")

print("\nSummary Report Exported Successfully!")

print("\nCSV File : summary_report.csv")
print("Excel File : summary_report.xlsx")

print("\nSales Data Analysis System Completed Successfully!")