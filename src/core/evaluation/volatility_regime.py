from src.core.evaluation.matrix_profile import MatrixProfile
from src.utils.logger import log_table, log_computation_info
import polars as pl
import numpy as np
import cupy as cp

def detect_volatility_regimes(time_series: pl.Series, window_length: int = 30, use_cuda: bool = False) -> pl.Series:
  """Label volatility regimes based on anomalies in rolling volatility."""

  if cp.cuda.runtime.getVersion() and not use_cuda:
    use_cuda = True
  
    computation_type = "CPU" if use_cuda else "CUDA"
    
    log_table("Matrix Profile Volatility Regime Shift Detection", "INFO", f"Starting computation with. {computation_type}")
    
  mp = MatrixProfile(time_series, window_length)
  mp.compute()

  log_table("Matrix Profile Volatility Regime Shift Detection", "INFO", f"Starting with window size: {window_length}")

  rolling_volatility = pl.Series(time_series).rolling_std(window_length).to_numpy()
  mp_vol = mp.calculate(rolling_volatility, window_length)
  profile_distances = mp_vol[:, 0]

  regimes = np.array(['normal'] * len(rolling_volatility))
  high_volatility = profile_distances > np.percentile(profile_distances, 90)
  low_volatility = profile_distances < np.percentile(profile_distances, 10)

  regimes[high_volatility] = 'high_volatility'
  regimes[low_volatility] = 'low_volatility'

  high_count = np.sum(high_volatility)
  low_count = np.sum(low_volatility)

  log_table("Volatility Regime Detection", "INFO", f"High volatility periods: {high_count}, Low volatility periods: {low_count}")

  return pl.Series(regimes)