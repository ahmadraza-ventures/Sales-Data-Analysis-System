# ============================================================
# SALES PREDICTION MODEL
# COMPLETE MACHINE LEARNING PROJECT
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("sales_data.csv")

print("=" * 60)
print("SALES PREDICTION MODEL")
print("=" * 60)

print("\nOriginal Dataset:")
print(df)


# ============================================================
# 2. DATA CLEANING
# ============================================================

# Convert price into numeric
df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

# Convert quantity into numeric
df["quantity"] = pd.to_numeric(
    df["quantity"],
    errors="coerce"
)

# Remove rows where price or quantity is missing
df = df.dropna(
    subset=["price", "quantity"]
).copy()

print("\nCleaned Dataset:")
print(df)


# ============================================================
# 3. CREATE REVENUE FEATURE
# ============================================================

df["revenue"] = (
    df["price"] * df["quantity"]
)

print("\nDataset with Revenue:")
print(df)


# ============================================================
# 4. CREATE HIGH SALE TARGET
# ============================================================

average_quantity = df["quantity"].mean()

df["high_sale"] = (
    df["quantity"] > average_quantity
).astype(int)

print("\nAverage Quantity:")
print(average_quantity)

print("\nHigh Sale Column:")
print(
    df[
        ["quantity", "high_sale"]
    ]
)


# ============================================================
# 5. ENCODE CATEGORICAL COLUMNS
# ============================================================

df_encoded = pd.get_dummies(
    df,
    columns=[
        "category",
        "region"
    ],
    drop_first=True,
    dtype=int
)

print("\nEncoded Dataset:")
print(df_encoded)


# ============================================================
# 6. SELECT ONLY REQUIRED FEATURES
# ============================================================
#
# IMPORTANT:
#
# We DON'T use:
# product
# order_id
# date
#
# We ONLY use:
# price
# quantity
# category
# region
#
# After encoding, category and region become
# numeric columns.
# ============================================================

feature_columns = [
    "price",
    "quantity"
]

# Add encoded category columns
for column in df_encoded.columns:

    if column.startswith("category_"):
        feature_columns.append(column)


# Add encoded region columns
for column in df_encoded.columns:

    if column.startswith("region_"):
        feature_columns.append(column)


print("\nFeatures Used By ML Models:")

for feature in feature_columns:
    print("-", feature)


# ============================================================
# 7. REGRESSION DATA
# ============================================================

X_reg = df_encoded[
    feature_columns
]

y_reg = df_encoded[
    "revenue"
]

print("\n" + "=" * 60)
print("REGRESSION DATA")
print("=" * 60)

print("\nX Regression:")
print(X_reg)

print("\ny Regression:")
print(y_reg)


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

print("\nRegression:")
print(
    "X_train Shape:",
    X_train.shape
)

print(
    "X_test Shape:",
    X_test.shape
)

print(
    "y_train Shape:",
    y_train.shape
)

print(
    "y_test Shape:",
    y_test.shape
)


# ============================================================
# 9. LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

# Predict revenue
y_pred = linear_model.predict(
    X_test
)


# ============================================================
# 10. REGRESSION EVALUATION
# ============================================================

mse = mean_squared_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("LINEAR REGRESSION RESULTS")
print("=" * 60)

print("\nActual Revenue:")
print(y_test.values)

print("\nPredicted Revenue:")
print(y_pred)

print("\nMean Squared Error:")
print(mse)

print("\nR2 Score:")
print(r2)


# ============================================================
# 11. CLASSIFICATION DATA
# ============================================================

X_class = df_encoded[
    feature_columns
]

y_class = df_encoded[
    "high_sale"
]

print("\n" + "=" * 60)
print("CLASSIFICATION DATA")
print("=" * 60)

print("\nClass Distribution:")
print(
    y_class.value_counts()
)


# ============================================================
# 12. TRAIN / TEST SPLIT FOR CLASSIFICATION
# ============================================================
#
# stratify remove kiya gaya hai because dataset bohat small hai.
# ============================================================

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class,
    y_class,
    test_size=0.2,
    random_state=42
)

print("\nClassification:")
print(
    "X_train Shape:",
    X_train_c.shape
)

print(
    "X_test Shape:",
    X_test_c.shape
)

print(
    "y_train Shape:",
    y_train_c.shape
)

print(
    "y_test Shape:",
    y_test_c.shape
)


# ============================================================
# 13. STANDARD SCALER
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_c
)

X_test_scaled = scaler.transform(
    X_test_c
)

print("\nFeatures Scaled Successfully.")


# ============================================================
# 14. LOGISTIC REGRESSION
# ============================================================

logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(
    X_train_scaled,
    y_train_c
)

# Prediction
logistic_pred = logistic_model.predict(
    X_test_scaled
)


# ============================================================
# 15. LOGISTIC REGRESSION EVALUATION
# ============================================================

logistic_accuracy = accuracy_score(
    y_test_c,
    logistic_pred
)

logistic_cm = confusion_matrix(
    y_test_c,
    logistic_pred
)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print("\nAccuracy:")
print(logistic_accuracy)

print("\nConfusion Matrix:")
print(logistic_cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test_c,
        logistic_pred,
        zero_division=0
    )
)


# ============================================================
# 16. RANDOM FOREST CLASSIFIER
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(
    X_train_c,
    y_train_c
)

# Prediction
rf_pred = random_forest.predict(
    X_test_c
)


# ============================================================
# 17. RANDOM FOREST EVALUATION
# ============================================================

rf_accuracy = accuracy_score(
    y_test_c,
    rf_pred
)

rf_cm = confusion_matrix(
    y_test_c,
    rf_pred
)

print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print("\nAccuracy:")
print(rf_accuracy)

print("\nConfusion Matrix:")
print(rf_cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test_c,
        rf_pred,
        zero_division=0
    )
)


# ============================================================
# 18. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    "\nLogistic Regression Accuracy:",
    logistic_accuracy
)

print(
    "Random Forest Accuracy:",
    rf_accuracy
)

if logistic_accuracy > rf_accuracy:

    print(
        "\nLogistic Regression performed better."
    )

elif rf_accuracy > logistic_accuracy:

    print(
        "\nRandom Forest performed better."
    )

else:

    print(
        "\nBoth models have the same accuracy."
    )


# ============================================================
# 19. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

feature_importance = (
    random_forest.feature_importances_
)

importance_df = pd.DataFrame({

    "Feature": feature_columns,

    "Importance": feature_importance

})


# Sort by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance_df
)


# ============================================================
# 20. FEATURE IMPORTANCE PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

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

plt.show()


# ============================================================
# 21. CROSS VALIDATION
# ============================================================
#
# Dataset bohat small hai.
#
# Hum available samples ke according CV folds choose
# kar rahe hain.
# ============================================================

class_counts = y_class.value_counts()

minimum_class_count = class_counts.min()

cv_folds = min(
    3,
    minimum_class_count
)

print("\n" + "=" * 60)
print("CROSS VALIDATION")
print("=" * 60)

print(
    "\nAvailable samples per class:"
)

print(class_counts)

print(
    "\nCV Folds:",
    cv_folds
)


# ============================================================
# 22. CROSS VALIDATION
# ============================================================

if cv_folds >= 2:

    # --------------------------------------------------------
    # Logistic Regression CV
    # --------------------------------------------------------

    logistic_cv_scores = cross_val_score(
        logistic_model,
        X_train_scaled,
        y_train_c,
        cv=cv_folds,
        scoring="accuracy"
    )

    print(
        "\nLogistic Regression CV Scores:"
    )

    print(
        logistic_cv_scores
    )

    print(
        "\nLogistic Regression Average CV Accuracy:"
    )

    print(
        logistic_cv_scores.mean()
    )


    # --------------------------------------------------------
    # Random Forest CV
    # --------------------------------------------------------

    rf_cv_scores = cross_val_score(
        random_forest,
        X_class,
        y_class,
        cv=cv_folds,
        scoring="accuracy"
    )

    print(
        "\nRandom Forest CV Scores:"
    )

    print(
        rf_cv_scores
    )

    print(
        "\nRandom Forest Average CV Accuracy:"
    )

    print(
        rf_cv_scores.mean()
    )

else:

    print(
        "\nCross Validation skipped."
    )

    print(
        "Dataset mein har class ke enough samples nahi hain."
    )


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("\nREGRESSION")

print(
    "Mean Squared Error:",
    mse
)

print(
    "R2 Score:",
    r2
)


print("\nCLASSIFICATION")

print(
    "Logistic Regression Accuracy:",
    logistic_accuracy
)

print(
    "Random Forest Accuracy:",
    rf_accuracy
)


print("\nFEATURES USED")

for feature in feature_columns:

    print(
        "-",
        feature
    )


print("\n" + "=" * 60)

print(
    "SALES PREDICTION MODEL COMPLETED SUCCESSFULLY!"
)

print("=" * 60)
