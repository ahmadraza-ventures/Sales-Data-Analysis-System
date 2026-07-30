# Sales-Data-Analysis-System

## Project Overview

This project is a Sales Data Analysis System developed in Python using Pandas and NumPy. It demonstrates real-world data cleaning, data analysis, revenue calculation, outlier detection, and report generation from a CSV dataset.

---

## Technologies Used

- Python
- Pandas
- NumPy
- OpenPyXL

---

## Dataset

The project uses a CSV file named:

```
sales_data.csv
```

Dataset Columns:

- order_id
- product
- category
- price
- quantity
- date
- region

The dataset contains intentionally messy data such as:

- Missing values
- Duplicate rows
- Incorrect data types

---

## Features

### 1. Data Cleaning

- Load CSV using Pandas
- Handle missing values
- Remove duplicate rows
- Convert price to numeric values
- Convert date column to datetime format

---

### 2. NumPy Operations

- Convert price and quantity columns to NumPy arrays
- Calculate revenue using vectorized operations
- Calculate:
  - Total Revenue
  - Mean Price
  - Median Price
  - Standard Deviation

---

### 3. Data Analysis

- Revenue by Category
- Revenue by Region
- Top 5 Best-Selling Products
- Month-wise Revenue Trend

---

### 4. Outlier Detection

Detect unusual prices using the IQR (Interquartile Range) method.

---

### 5. Summary Report

Generate a summary report containing:

- Category
- Total Revenue
- Average Order Value
- Number of Orders

---

### 6. Export Reports

The project exports:

- summary_report.csv
- summary_report.xlsx

---

## Project Structure

```
Sales-Data-Analysis-System/
│
├── sales_analysis.py
├── sales_data.csv
├── summary_report.csv
├── summary_report.xlsx
└── README.md
```

---

## How to Run

1. Install required libraries

```bash
pip install pandas numpy openpyxl
```

2. Run the project

```bash
python sales_analysis.py
```

---

## Output

The program performs data cleaning, analysis, revenue calculations, outlier detection, and generates summary reports in both CSV and Excel formats.

---

## Learning Outcomes

Through this project, I learned:

- CSV File Handling
- Data Cleaning with Pandas
- NumPy Vectorized Operations
- GroupBy Analysis
- Revenue Calculation
- Outlier Detection using IQR
- Report Generation
- CSV and Excel Export

---

## Author

**Ahmad Raza**
