# ============================================================
# SALES PREDICTION MODEL
# PART 1 - DATASET PREPARATION
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("sales_data.csv")

print("=" * 60)
print("SALES PREDICTION MODEL - PART 1")
print("=" * 60)

print("\nOriginal Dataset:")
print(df)


# ============================================================
# 2. CHECK DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

df.info()


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())


# ============================================================
# 4. CONVERT PRICE TO NUMERIC
# ============================================================

df["price"] = (
    df["price"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)


# ============================================================
# 5. CONVERT QUANTITY TO NUMERIC
# ============================================================

df["quantity"] = pd.to_numeric(
    df["quantity"],
    errors="coerce"
)


# ============================================================
# 6. HANDLE MISSING / INVALID PRICE
# ============================================================

price_median = df["price"].median()

df["price"] = df["price"].fillna(
    price_median
)


# ============================================================
# 7. HANDLE MISSING QUANTITY
# ============================================================

quantity_median = df["quantity"].median()

df["quantity"] = df["quantity"].fillna(
    quantity_median
)


# ============================================================
# 8. REMOVE DUPLICATE ROWS
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

print("Rows before removing duplicates:", before_duplicates)
print("Rows after removing duplicates :", after_duplicates)


# ============================================================
# 9. CONVERT DATE TO DATETIME
# ============================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# ============================================================
# 10. HANDLE MISSING DATE
# ============================================================

date_mode = df["date"].mode()[0]

df["date"] = df["date"].fillna(
    date_mode
)


# ============================================================
# 11. CREATE REVENUE FEATURE
#
# Revenue = Price × Quantity
# ============================================================

df["revenue"] = (
    df["price"] * df["quantity"]
)


print("\n" + "=" * 60)
print("REVENUE FEATURE")
print("=" * 60)

print(
    df[
        ["price", "quantity", "revenue"]
    ]
)


# ============================================================
# 12. ENCODE CATEGORY AND REGION
#
# Machine Learning cannot directly understand text categories.
# pd.get_dummies() converts them into numerical columns.
# ============================================================

df = pd.get_dummies(
    df,
    columns=[
        "category",
        "region"
    ],
    drop_first=True,
    dtype=int
)


print("\n" + "=" * 60)
print("DATASET AFTER ENCODING")
print("=" * 60)

print(df)


# ============================================================
# 13. CREATE FEATURES (X)
#
# Features:
# price
# quantity
# encoded category
# encoded region
# ============================================================

X = df.drop(
    columns=[
        "revenue",
        "order_id",
        "product",
        "date"
    ]
)


# ============================================================
# 14. CREATE TARGET (y)
#
# Target = Revenue
# ============================================================

y = df["revenue"]


# ============================================================
# 15. DISPLAY FEATURES
# ============================================================

print("\n" + "=" * 60)
print("FEATURES (X)")
print("=" * 60)

print(X)

print("\nFeature Columns:")
print(X.columns.tolist())


# ============================================================
# 16. DISPLAY TARGET
# ============================================================

print("\n" + "=" * 60)
print("TARGET (y) - REVENUE")
print("=" * 60)

print(y)


# ============================================================
# 17. CHECK DATA SHAPE
# ============================================================

print("\n" + "=" * 60)
print("DATASET SHAPE")
print("=" * 60)

print("Dataset Shape:", df.shape)
print("X Shape:", X.shape)
print("y Shape:", y.shape)


# ============================================================
# 18. FINAL MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 60)
print("FINAL MISSING VALUE CHECK")
print("=" * 60)

print("Missing values in X:")
print(X.isnull().sum())

print("\nMissing values in y:")
print(y.isnull().sum())


# ============================================================
# PART 1 COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("PART 1 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nDataset is now ready for Machine Learning.")
print("X = Features")
print("y = Revenue Target")















# ============================================================
# PART 2 - REGRESSION MODEL
# Train/Test Split + Linear Regression
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. SPLIT DATA INTO TRAINING AND TESTING DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("Training Features (X_train):", X_train.shape)
print("Testing Features (X_test):", X_test.shape)
print("Training Target (y_train):", y_train.shape)
print("Testing Target (y_test):", y_test.shape)


# ============================================================
# 2. CREATE LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# 3. TRAIN THE MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


print("\n" + "=" * 60)
print("LINEAR REGRESSION")
print("=" * 60)

print("Model trained successfully!")


# ============================================================
# 4. PREDICT REVENUE
# ============================================================

y_pred = model.predict(
    X_test
)


print("\n" + "=" * 60)
print("REVENUE PREDICTIONS")
print("=" * 60)

print("Actual Revenue:")
print(y_test.values)

print("\nPredicted Revenue:")
print(y_pred)


# ============================================================
# 5. MEAN SQUARED ERROR
# ============================================================

mse = mean_squared_error(
    y_test,
    y_pred
)


# ============================================================
# 6. R² SCORE
# ============================================================

r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# 7. MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)


# ============================================================
# 8. MODEL COEFFICIENTS
# ============================================================

print("\n" + "=" * 60)
print("FEATURE COEFFICIENTS")
print("=" * 60)

for feature, coefficient in zip(
    X.columns,
    model.coef_
):
    print(
        f"{feature}: {coefficient:.2f}"
    )


# ============================================================
# 9. INTERCEPT
# ============================================================

print("\nIntercept:", model.intercept_)


# ============================================================
# PART 2 COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("PART 2 COMPLETED SUCCESSFULLY!")
print("=" * 60)

















# ============================================================
# PART 3 - CLASSIFICATION MODEL
# High Sale Prediction using Logistic Regression
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. CALCULATE AVERAGE QUANTITY
# ============================================================

average_quantity = df["quantity"].mean()

print("\n" + "=" * 60)
print("AVERAGE QUANTITY")
print("=" * 60)

print("Average Quantity:", average_quantity)


# ============================================================
# 2. CREATE HIGH_SALE TARGET
#
# quantity > average → 1
# quantity <= average → 0
# ============================================================

df["high_sale"] = (
    df["quantity"] > average_quantity
).astype(int)


print("\n" + "=" * 60)
print("HIGH SALE TARGET")
print("=" * 60)

print(
    df[
        ["quantity", "high_sale"]
    ]
)


# ============================================================
# 3. CREATE CLASSIFICATION FEATURES
#
# We remove:
# revenue   → not needed for high_sale prediction
# high_sale → target
# order_id  → identifier
# product   → text
# date      → not being used
# ============================================================

X_classification = df.drop(
    columns=[
        "revenue",
        "high_sale",
        "order_id",
        "product",
        "date"
    ]
)


# ============================================================
# 4. CREATE CLASSIFICATION TARGET
# ============================================================

y_classification = df["high_sale"]


print("\n" + "=" * 60)
print("CLASSIFICATION DATA")
print("=" * 60)

print("X Classification:")
print(X_classification)

print("\nY Classification:")
print(y_classification)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_classification,
    y_classification,
    test_size=0.2,
    random_state=42,
    stratify=y_classification
)


print("\n" + "=" * 60)
print("CLASSIFICATION TRAIN / TEST SPLIT")
print("=" * 60)

print("Training Data:", X_train_class.shape)
print("Testing Data :", X_test_class.shape)


# ============================================================
# 6. CREATE LOGISTIC REGRESSION MODEL
# ============================================================

classification_model = LogisticRegression(
    max_iter=1000
)


# ============================================================
# 7. TRAIN CLASSIFICATION MODEL
# ============================================================

classification_model.fit(
    X_train_class,
    y_train_class
)


print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

print("Classification model trained successfully!")


# ============================================================
# 8. MAKE PREDICTIONS
# ============================================================

y_class_pred = classification_model.predict(
    X_test_class
)


print("\n" + "=" * 60)
print("HIGH SALE PREDICTIONS")
print("=" * 60)

print("Actual High Sale:")
print(y_test_class.values)

print("\nPredicted High Sale:")
print(y_class_pred)


# ============================================================
# 9. CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test_class,
    y_class_pred
)


print("\n" + "=" * 60)
print("ACCURACY")
print("=" * 60)

print("Accuracy:", accuracy)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test_class,
    y_class_pred
)


print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test_class,
        y_class_pred,
        zero_division=0
    )
)


# ============================================================
# PART 3 COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("PART 3 COMPLETED SUCCESSFULLY!")
print("=" * 60)











# ============================================================
# PART 4 - ADVANCED CLASSIFICATION
# StandardScaler + Random Forest + Feature Importance
# ============================================================

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# ============================================================
# 1. STANDARD SCALER
# ============================================================

scaler = StandardScaler()


# Fit scaler on training data
X_train_scaled = scaler.fit_transform(
    X_train_class
)


# Transform testing data
X_test_scaled = scaler.transform(
    X_test_class
)


print("\n" + "=" * 60)
print("STANDARD SCALER")
print("=" * 60)

print("Features have been standardized successfully!")


# ============================================================
# 2. LOGISTIC REGRESSION WITH SCALED DATA
# ============================================================

scaled_logistic_model = LogisticRegression(
    max_iter=1000
)

scaled_logistic_model.fit(
    X_train_scaled,
    y_train_class
)


# Predict using scaled test data
y_scaled_logistic_pred = (
    scaled_logistic_model.predict(
        X_test_scaled
    )
)


# Calculate accuracy
scaled_logistic_accuracy = accuracy_score(
    y_test_class,
    y_scaled_logistic_pred
)


print("\n" + "=" * 60)
print("SCALED LOGISTIC REGRESSION")
print("=" * 60)

print(
    "Logistic Regression Accuracy:",
    scaled_logistic_accuracy
)


# ============================================================
# 3. RANDOM FOREST CLASSIFIER
# ============================================================

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ============================================================
# 4. TRAIN RANDOM FOREST
# ============================================================

random_forest_model.fit(
    X_train_class,
    y_train_class
)


print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

print("Random Forest model trained successfully!")


# ============================================================
# 5. RANDOM FOREST PREDICTIONS
# ============================================================

y_rf_pred = random_forest_model.predict(
    X_test_class
)


print("\nActual High Sale:")
print(y_test_class.values)

print("\nRandom Forest Predictions:")
print(y_rf_pred)


# ============================================================
# 6. RANDOM FOREST ACCURACY
# ============================================================

random_forest_accuracy = accuracy_score(
    y_test_class,
    y_rf_pred
)


print("\n" + "=" * 60)
print("RANDOM FOREST ACCURACY")
print("=" * 60)

print(
    "Random Forest Accuracy:",
    random_forest_accuracy
)


# ============================================================
# 7. COMPARE LOGISTIC REGRESSION
#    AND RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("MODEL ACCURACY COMPARISON")
print("=" * 60)

print(
    f"Logistic Regression Accuracy: "
    f"{scaled_logistic_accuracy:.2f}"
)

print(
    f"Random Forest Accuracy: "
    f"{random_forest_accuracy:.2f}"
)


if random_forest_accuracy > scaled_logistic_accuracy:

    print(
        "\nRandom Forest performed better."
    )

elif random_forest_accuracy < scaled_logistic_accuracy:

    print(
        "\nLogistic Regression performed better."
    )

else:

    print(
        "\nBoth models have the same accuracy."
    )


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

feature_importance = (
    random_forest_model.feature_importances_
)


# ============================================================
# 9. CREATE FEATURE IMPORTANCE DATAFRAME
# ============================================================

importance_df = pd.DataFrame({
    "Feature": X_classification.columns,
    "Importance": feature_importance
})


# Sort from highest to lowest
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(importance_df)


# ============================================================
# 10. PLOT FEATURE IMPORTANCE
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.title(
    "Random Forest Feature Importance"
)

plt.xlabel(
    "Features"
)

plt.ylabel(
    "Importance"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()


# ============================================================
# 11. SAVE FEATURE IMPORTANCE GRAPH
# ============================================================

plt.savefig(
    "feature_importance.png"
)

plt.show()


# ============================================================
# PART 4 COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("PART 4 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(
    "\nFeature importance graph saved as:"
)

print("feature_importance.png")



















# ============================================================
# PART 5 - CROSS VALIDATION
# Check Model Stability
# ============================================================

from sklearn.model_selection import cross_val_score


# ============================================================
# 1. CROSS VALIDATION - LOGISTIC REGRESSION
# ============================================================

logistic_cv_scores = cross_val_score(
    scaled_logistic_model,
    X_train_scaled,
    y_train_class,
    cv=3,
    scoring="accuracy"
)


print("\n" + "=" * 60)
print("LOGISTIC REGRESSION - CROSS VALIDATION")
print("=" * 60)

print("Cross Validation Scores:")
print(logistic_cv_scores)

print(
    "Mean CV Accuracy:",
    logistic_cv_scores.mean()
)


# ============================================================
# 2. CROSS VALIDATION - RANDOM FOREST
# ============================================================

random_forest_cv_scores = cross_val_score(
    random_forest_model,
    X_train_class,
    y_train_class,
    cv=3,
    scoring="accuracy"
)


print("\n" + "=" * 60)
print("RANDOM FOREST - CROSS VALIDATION")
print("=" * 60)

print("Cross Validation Scores:")
print(random_forest_cv_scores)

print(
    "Mean CV Accuracy:",
    random_forest_cv_scores.mean()
)


# ============================================================
# 3. FINAL MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    f"Logistic Regression Test Accuracy: "
    f"{scaled_logistic_accuracy:.2f}"
)

print(
    f"Random Forest Test Accuracy: "
    f"{random_forest_accuracy:.2f}"
)

print(
    f"Logistic Regression CV Accuracy: "
    f"{logistic_cv_scores.mean():.2f}"
)

print(
    f"Random Forest CV Accuracy: "
    f"{random_forest_cv_scores.mean():.2f}"
)


# ============================================================
# 4. DETERMINE BEST MODEL USING CV
# ============================================================

if (
    random_forest_cv_scores.mean()
    >
    logistic_cv_scores.mean()
):

    print(
        "\nRandom Forest has better "
        "cross-validation performance."
    )

elif (
    random_forest_cv_scores.mean()
    <
    logistic_cv_scores.mean()
):

    print(
        "\nLogistic Regression has better "
        "cross-validation performance."
    )

else:

    print(
        "\nBoth models have the same "
        "cross-validation performance."
    )


# ============================================================
# 5. CROSS VALIDATION STABILITY
# ============================================================

print("\n" + "=" * 60)
print("MODEL STABILITY")
print("=" * 60)

print(
    "Logistic Regression CV Standard Deviation:",
    logistic_cv_scores.std()
)

print(
    "Random Forest CV Standard Deviation:",
    random_forest_cv_scores.std()
)


# ============================================================
# PART 5 COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("PART 5 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nSALES PREDICTION PROJECT COMPLETED!")