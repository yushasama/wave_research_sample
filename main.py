from src.core.evaluation.matrix_profile import MatrixProfile
from src.core.models.factor_models.base_factor_model import BaseFactorModel
from sklearn.preprocessing import StandardScaler
import polars as pl
import numpy as np
import torch
import time

# Load training data
train_data = pl.read_csv("factor_train_data.csv")

# Clean and convert data types
train_data = train_data.with_columns(
    [pl.col(col).cast(pl.Utf8) for col in train_data.columns if col != "DATE"]
)

train_data = train_data.with_columns(
    [pl.col(col).str.strip_chars() for col in train_data.columns if col != "DATE"]
)

train_data = train_data.with_columns(
    [pl.col(col).cast(pl.Float64) for col in train_data.columns if col != "DATE"]
)

# Standardize the features (excluding DATE and TARGET columns)
scaler = StandardScaler()
feature_columns = [col for col in train_data.columns if col not in ["DATE", "TARGET"]]
train_features = train_data.select(feature_columns).to_numpy()
standardized_features = scaler.fit_transform(train_features)

# Update the training data with standardized features
for idx, col in enumerate(feature_columns):
    train_data = train_data.with_columns(pl.Series(name=col, values=standardized_features[:, idx]))

# Initialize and train the model
model = BaseFactorModel("SPY Factor Model", train_data, use_cuda=True)

for factor_name in feature_columns:
    model.add_factor(factor_name, lambda data=train_data, col=factor_name: data[col])

model.fit(target_column="TARGET", model_type="ridge", alpha=1.0)

# Load and preprocess test data
test_data = pl.read_csv("factor_test_data.csv")

test_data = test_data.with_columns(
    [pl.col(col).cast(pl.Utf8) for col in test_data.columns if col != "DATE"]
)

test_data = test_data.with_columns(
    [pl.col(col).str.strip_chars() for col in test_data.columns if col != "DATE"]
)

test_data = test_data.with_columns(
    [pl.col(col).cast(pl.Float64) for col in test_data.columns if col not in ["DATE", "TARGET"]]
)

# Standardize test features using the same scaler
test_features = test_data.select(feature_columns).to_numpy()
standardized_test_features = scaler.transform(test_features)

# Update the test data with standardized features
for idx, col in enumerate(feature_columns):
    test_data = test_data.with_columns(pl.Series(name=col, values=standardized_test_features[:, idx]))

# Make predictions
predictions = model.predict(test_data.select(feature_columns))

print(predictions)