
# SALES DATA DASHBOARD SYSTEM
# Part 1
# Import Libraries + Load Dataset + Data Cleaning

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Load Dataset


try:
    df = pd.read_csv("sales_data.csv")
    print("=" * 60)
    print("Sales Dataset Loaded Successfully!")
    print("=" * 60)

except FileNotFoundError:
    print("Error: sales_data.csv not found.")
    exit()


# Display Dataset

print("\nFirst 5 Rows")
print(df.head())

print("\nLast 5 Rows")
print(df.tail())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())


# Data Cleaning

# Convert Date Column

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

# Convert Price into Numeric

df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

# Convert Quantity into Numeric

df["quantity"] = pd.to_numeric(
    df["quantity"],
    errors="coerce"
)

# Remove Missing Values

df.dropna(
    subset=[
        "date",
        "price",
        "quantity"
    ],
    inplace=True
)

# Remove Duplicate Records

duplicates = df.duplicated().sum()

print("\nDuplicate Rows :", duplicates)

df.drop_duplicates(inplace=True)

# Reset Index

df.reset_index(
    drop=True,
    inplace=True
)

print("\nDataset After Cleaning")

print(df.head())


# Revenue Calculation


df["revenue"] = (
    df["price"] *
    df["quantity"]
)

print("\nRevenue Column Created Successfully")

print(
    df[
        [
            "product",
            "price",
            "quantity",
            "revenue"
        ]
    ].head()
)


# Basic Statistics


print("\nBasic Statistics")

print(df.describe())

print("\nTotal Revenue")

print(df["revenue"].sum())

print("\nAverage Revenue")

print(df["revenue"].mean())

print("\nHighest Revenue")

print(df["revenue"].max())

print("\nLowest Revenue")

print(df["revenue"].min())

print("\nRevenue Column Information")

print(df["revenue"].describe())

print("\nPart 1 Completed Successfully!")

print("=" * 60)


# PART 2
# Feature Engineering + Dashboard


# Month Feature


df["month"] = df["date"].dt.strftime("%b")
df["month_number"] = df["date"].dt.month

print("\nMonth Column")

print(df[["date", "month"]].head())

# Revenue by Category


category_sales = (
    df.groupby("category")["revenue"]
      .sum()
)

print("\nRevenue by Category")

print(category_sales)


# Revenue by Region


region_sales = (
    df.groupby("region")["revenue"]
      .sum()
)

print("\nRevenue by Region")

print(region_sales)


# Monthly Revenue


monthly_sales = (
    df.groupby(["month_number", "month"])["revenue"]
      .sum()
      .reset_index()
      .sort_values("month_number")
)

print("\nMonthly Revenue")

print(monthly_sales)


# Dashboard (2 × 2)


fig, axes = plt.subplots(
    2,
    2,
    figsize=(14,10)
)

fig.suptitle(
    "Sales Dashboard",
    fontsize=18,
    fontweight="bold"
)


# Chart 1
# Revenue by Category
# 

axes[0,0].bar(
    category_sales.index,
    category_sales.values,
    color=["red","blue","green","orange","purple"]
)

axes[0,0].set_title("Revenue by Category")
axes[0,0].set_xlabel("Category")
axes[0,0].set_ylabel("Revenue")
axes[0,0].legend(["Revenue"])
axes[0,0].tick_params(axis="x", rotation=45)


# Chart 2
# Sales Distribution by Region


axes[0,1].pie(
    region_sales.values,
    labels=region_sales.index,
    autopct="%1.1f%%",
    colors=[
        "gold",
        "lightblue",
        "lightgreen",
        "pink",
        "orange"
    ]
)

axes[0,1].set_title("Sales Distribution by Region")


# Chart 3
# Monthly Revenue


axes[1,0].plot(
    monthly_sales["month"],
    monthly_sales["revenue"],
    marker="o",
    color="blue",
    linewidth=2,
    label="Revenue"
)

axes[1,0].set_title("Month-wise Revenue")
axes[1,0].set_xlabel("Month")
axes[1,0].set_ylabel("Revenue")
axes[1,0].legend()
axes[1,0].grid(True)


# Chart 4
# Histogram

axes[1,1].hist(
    df["price"],
    bins=10,
    color="skyblue",
    edgecolor="black",
    label="Order Prices"
)

axes[1,1].set_title("Distribution of Order Prices")
axes[1,1].set_xlabel("Price")
axes[1,1].set_ylabel("Frequency")
axes[1,1].legend()

# Save Dashboard


plt.tight_layout()

plt.savefig("sales_dashboard.png")

plt.show()

print("\nDashboard Saved Successfully!")

print("=" * 60)
print("Part 2 Completed Successfully!")
print("=" * 60)


# PART 3
# Scatter Plot + Outliers + Stacked Bar Chart



# Detect Outliers


# Using IQR Method

Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - (1.5 * IQR)
upper_limit = Q3 + (1.5 * IQR)

normal_data = df[
    (df["price"] >= lower_limit) &
    (df["price"] <= upper_limit)
]

outliers = df[
    (df["price"] < lower_limit) |
    (df["price"] > upper_limit)
]

print("\nTotal Outliers Found :", len(outliers))


# Scatter Plot


plt.figure(figsize=(8,6))

# Normal Points

plt.scatter(
    normal_data["price"],
    normal_data["quantity"],
    color="blue",
    alpha=0.7,
    label="Normal Data"
)

# Outliers

plt.scatter(
    outliers["price"],
    outliers["quantity"],
    color="red",
    s=100,
    label="Outliers"
)

plt.title("Price vs Quantity")

plt.xlabel("Price")
plt.ylabel("Quantity")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig("scatter_outliers.png")

plt.show()

print("Scatter Plot Saved Successfully!")


# Revenue by Category & Region


stacked_data = df.pivot_table(
    values="revenue",
    index="category",
    columns="region",
    aggfunc="sum",
    fill_value=0
)

print("\nRevenue by Category and Region")

print(stacked_data)


# Stacked Bar Chart


plt.figure(figsize=(10,6))

stacked_data.plot(
    kind="bar",
    stacked=True
)

plt.title("Revenue by Category (Split by Region)")

plt.xlabel("Category")

plt.ylabel("Revenue")

plt.xticks(rotation=45)

plt.legend(title="Region")

plt.tight_layout()

plt.savefig("stacked_bar.png")

plt.show()

print("Stacked Bar Chart Saved Successfully!")

print("=" * 60)
print("Part 3 Completed Successfully!")
print("=" * 60)



# PART 4
# Summary Table + PDF Report + Project Summary



# Summary Table

summary_table = (
    df.groupby("category")
      .agg(
          Total_Revenue=("revenue", "sum"),
          Total_Orders=("order_id", "count"),
          Average_Price=("price", "mean"),
          Total_Quantity=("quantity", "sum")
      )
      .reset_index()
)

print("\nSummary Table")
print(summary_table)


# Create PDF Report


with PdfPages("Sales_Report.pdf") as pdf:

    
    # Dashboard Image
    
    dashboard = plt.imread("sales_dashboard.png")

    fig = plt.figure(figsize=(10,7))

    plt.imshow(dashboard)

    plt.axis("off")

    plt.title("Sales Dashboard")

    pdf.savefig(fig)

    plt.close()


    
    # Scatter Plot Image
    

    scatter = plt.imread("scatter_outliers.png")

    fig = plt.figure(figsize=(8,6))

    plt.imshow(scatter)

    plt.axis("off")

    plt.title("Scatter Plot")

    pdf.savefig(fig)

    plt.close()


   
    # Stacked Bar Image
    

    stacked = plt.imread("stacked_bar.png")

    fig = plt.figure(figsize=(8,6))

    plt.imshow(stacked)

    plt.axis("off")

    plt.title("Revenue by Category (Split by Region)")

    pdf.savefig(fig)

    plt.close()


    # Summary Table
  

    fig, ax = plt.subplots(figsize=(10,4))

    ax.axis("off")

    table = ax.table(
        cellText=summary_table.values,
        colLabels=summary_table.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title("Sales Summary")

    pdf.savefig(fig)

    plt.close()

print("\nSales_Report.pdf Created Successfully!")


# Final Project Summary

print("\n" + "=" * 60)
print("FINAL PROJECT SUMMARY")
print("=" * 60)

print(f"Total Records       : {len(df)}")
print(f"Total Revenue       : {df['revenue'].sum():,.2f}")
print(f"Average Price       : {df['price'].mean():.2f}")
print(f"Average Quantity    : {df['quantity'].mean():.2f}")
print(f"Highest Price       : {df['price'].max():.2f}")
print(f"Lowest Price        : {df['price'].min():.2f}")

print("=" * 60)

print("\nGenerated Files")

print("✔ sales_dashboard.png")
print("✔ scatter_outliers.png")
print("✔ stacked_bar.png")
print("✔ Sales_Report.pdf")

print("\nProject Completed Successfully!")