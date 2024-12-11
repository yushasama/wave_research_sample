from sklearn.linear_model import LinearRegression, Ridge, Lasso
from src.core.evaluation.matrix_profile import MatrixProfile
from sklearn.preprocessing import StandardScaler
from src.utils.logger import log_table
from sklearn.decomposition import PCA
import polars as pl
import numpy as np
import logging

# Logging setup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseFactorModel:
  def __init__(self, model_name: str ,data: pl.DataFrame, use_cuda: bool = False):
    """
    Initialize the factor model with historical data.

    Parameters:
      data (pl.DataFrame): Historical value and factor data.
    """
    
    self.model_name = model_name
    self.data = data
    self.factors = {}
    self.model = None
    self.fmps = {}
    self.use_cuda = use_cuda

    log_table(f"{self.model_name} Factor Model Initialization", "INFO", f"{self.model_name} initialized successfully.")


  def add_factor(self, factor_name: str, calculation_func):
    """
    Add a custom factor to the model.

    Parameters:
        factor_name (str): The name of the factor.
        calculation_func (callable): A function that calculates the factor.
    """

    calculated_factor = calculation_func(self.data)

    if not isinstance(calculated_factor, pl.Series):
      log_table(f"{self.model_name} Factor Model Addition", "ERROR", f"The calculation function must return a 'pl.Series'. Got: {type(calculated_factor)}", level="ERROR")
    
    # Step 1: Calculate the new factor using the provided calculation function
    new_factor = calculation_func(self.data)
    
    # Step 2: Orthogonalize the new factor against existing factors
    existing_factors = pl.DataFrame(self.factors)

    if not existing_factors.is_empty():
        # Perform linear regression to orthogonalize
        model = LinearRegression()
        model.fit(existing_factors.to_numpy(), new_factor.to_numpy())
        orthogonalized_factor = new_factor - model.predict(existing_factors.to_numpy())
        
    else:
        orthogonalized_factor = new_factor  # No existing factors to orthogonalize against

    # Step 3: Update the risk matrix with the new orthogonalized factor
    self.factors[factor_name] = orthogonalized_factor
    log_table(f"{self.model_name} Factor Model Addition", "INFO", f"Factor '{factor_name}' added after orthogonalization.")

    # Step 4: Create Factor-Mimicking Portfolio (FMP)
    self.create_fmp(factor_name, orthogonalized_factor)

    log_table(f"{self.model_name} Factor Model Addition", "INFO", f"Factor '{factor_name}' added and FMP created.")
  
  def create_fmp(self, factor_name: str, orthogonalized_factor: pl.Series):
    """
    Create a Factor-Mimicking Portfolio for the given factor.

    Parameters:
        factor_name (str): The name of the factor for which to create the FMP.
        orthogonalized_factor (pl.Series): The orthogonalized factor.
    """

    orthogonalized_factor = orthogonalized_factor.to_numpy()

    explained_variance = np.var(orthogonalized_factor)

    # Normalize weights
    weights = orthogonalized_factor / np.sum(orthogonalized_factor)

    weights *= explained_variance

    self.fmps[f"{factor_name}_FMP"] = weights
    log_table(f"{self.model_name} Factor Model FMP Creation", "INFO", f"FMP for factor '{factor_name}' created with weights.")

  def _add_pca_factor(self, num_components: int = 2):
    """
    Add PCA factors to the model based on existing factors.

    Parameters:
        num_components (int): Number of principal components to generate.
    """
    log_table(f"{self.model_name} PCA Factor Addition", "INFO", "Calculating PCA for existing factors.")

    if not self.factors:
      log_table(f"{self.model_name} PCA Factor Addition", "ERROR", "Factors not loaded.", level="ERROR")
      return

    # Standardize the data
    features = StandardScaler().fit_transform(pl.DataFrame(self.factors).to_numpy())

    # Perform PCA
    pca = PCA(n_components=num_components)
    pca_features = pca.fit_transform(features)

    # Add PCA factors to the model
    for i in range(num_components):
      self.add_factor(f"pca_factor_{i + 1}", lambda x: pca_features[:, i])

    log_table(f"{self.model_name} PCA Factor Addition", "INFO", f"Added {num_components} PCA factors.")

  # INCOMPLETE NEEDS MORE RESEARCH
  def dynamic_factor_loading(self):
    """
    Adjust the weights of the factors dynamically based on their variance explained.
    """
    log_table(f"{self.model_name} Dynamic Factor Weights", "INFO", "Calculating dynamic factor weights.")

    # Fit a regression model to determine factor importance based on variance explained
    X = pl.DataFrame(self.factors)
    y = self.data["target"]

    model = LinearRegression()
    model.fit(X.to_numpy(), y.to_numpy())

    # Get the coefficients as weights and normalize
    self.dynamic_weights = model.coef_ / np.sum(model.coef_)  # Normalize weights

    for factor_name, weight in zip(self.factors.keys(), self.dynamic_weights):
      log_table(f"{self.model_name} Dynamic Factor Weights", "INFO", f"Dynamic weight for '{factor_name}' adjusted to {weight:.4f}.")


  def fit(self, target_column: str, model_type: str = 'linear', model_func: callable = None, use_pca: bool = False, num_pca_components: int = None, dynamic_weights: bool = False, alpha: float = 1.0):
    """
    Fit the model using the factors and the target variable.

    Parameters:
      target_column (str): The name of the target variable (e.g., returns).
      model_type (str): Type of regression model to use ('linear', 'ridge', 'lasso', 'random_forest').
    """

    # Logging shapes
    log_table(f"{self.model_name} Factor Model Fitting", "INFO", f"data shape: {self.data.shape}")

    if use_pca:
      if not num_pca_components:
        num_pca_components = 2

      log_table(f"{self.model_name} Factor Model Fitting", "INFO", f"Adding PCA Factors, Num of PCA Components: {num_pca_components}")

      self._add_pca_factors(num_components=num_pca_components)

    X = pl.DataFrame(self.factors)
    y = self.data[target_column]

    model_mapping = {
      'linear': LinearRegression,
      'ridge': lambda: Ridge(alpha=alpha),
      'lasso': Lasso,
    }

    if model_func:
      self.model = model_func()
      log_table(f"{self.model_name} Factor Model Fitting", "INFO", f"Using custom model function provided by user. Model type: '{model_type}'.")

    elif model_type in model_mapping:
      self.model = model_mapping[model_type]()
      log_table(f"{self.model_name} Factor Model Fitting", "INFO", f"Using predefined model type: {model_type}.")

    else:
      log_table(f"{self.model_name} Factor Model Fitting", "Error", "Invalid model type. Choose from 'linear', 'ridge', 'lasso', 'random_forest'.", level="ERROR")
    
    self.model.fit(X.to_numpy(), y.to_numpy())
    log_table(f"{self.model_name} Factor Model Fitting", "INFO", "Model fitting complete.")


  def purged_cross_validation_matrix_profile(self, target_column: str, n_splits: int = 5, window_length: int = 30, model_type: str = 'linear', model_func: callable = None) -> None:
    """
    Perform purged cross-validation to evaluate model performance with matrix profile assistance.

    Parameters:
        target_column (str): The name of the target variable.
        n_splits (int): Number of splits for cross-validation.
    """

    log_table(f"{self.model_name} Purged Cross-Validation", "INFO", "Starting purged cross-validation.")

    fold_size = len(self.data) // n_splits

    for i in range(n_splits):
      
      # Create training and test sets

      train_data = pl.concat([self.data[:i * fold_size], self.data[(i + 1) * fold_size:]])
      test_data = self.data[i * fold_size:(i + 1) * fold_size]

      # Initialize MatrixProfile with the training data
      mp = MatrixProfile(train_data[list(self.factors.keys())].to_numpy(), window_length=window_length)
      mp.compute(use_cuda=self.use_cuda)

      # Find motifs and discords for the training data
      motifs = mp.find_motifs()  # Find all motifs
      discords = mp.find_discords()  # Find all discords

      # Fit the model using the training data
      self.fit(target_column=target_column, model_type=model_type, model_func=model_func)

      # Log validation performance
      log_table(f"{self.model_name} Purged Cross-Validation", "INFO", f"Completed validation for fold {i + 1}: Motifs: {motifs}, Discords: {discords}")

    log_table(f"{self.model_name} Purged Cross-Validation", "INFO", "Purged cross-validation complete.")
  

  def predict(self, test_data: pl.DataFrame) -> pl.Series:
    """
    Predict returns based on new data using the fitted model.

    Parameters:
        test_data (pl.DataFrame): DataFrame containing the factors for prediction.

    Returns:
        pl.Series: Predicted returns.
    """

    # Logging shape
    log_table(f"{self.model_name} Factor Model Prediction", "INFO", f"test data shape: {test_data.shape}")

    if self.model is None:
      log_table(f"{self.model_name} Factor Model Prediction", "Error", "Model has not been fitted yet. Call the 'fit' method first.", level="ERROR")

    predictions = self.model.predict(test_data.to_numpy())

    if len(predictions) == 0:
      log_table(f"{self.model_name} Factor Model Prediction", "WARNING", "Predictions returned None or empty.", level="ERROR")
      return None
    
    log_table(f"{self.model_name} Factor Model Prediction", "INFO", "Predictions made successfully.")

    return pl.DataFrame(predictions).to_numpy().astype(float)