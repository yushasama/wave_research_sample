from src.utils.logger import log_table, log_computation_info
import matplotlib.pyplot as plt
from typing import Union, List
from tabulate import tabulate
import polars as pl
import numpy as np
import cupy as cp
import cpuinfo
import logging
import torch

class MatrixProfile:
  def __init__(self, time_series: Union[List, np.ndarray], window_length: int):
    """
    Initialize the MatrixProfile.

    Parameters:
        time_series (List or np.ndarray): The time series data as a list or NumPy array.
        window_length (int): The length of the subsequences.
    """

    if len(time_series) < window_length:
      log_table("Matrix Profile Initialization", "ERROR", "Time series length must be greater than or equal to the window length.", level="ERROR")
      return

    self.time_series = pl.DataFrame({"values": time_series})
    self.window_length = window_length
    self.profile_distances = None
    self.profile_indices = None

    log_table("Initialization", "INFO", "MatrixProfile initialized successfully.", level="INFO")
  

  def _compute_with_cpu(self) -> None:
    """Compute matrix profile using CPU (NumPy)."""

    time_series_np= self.time_series.to_numpy()

    n = len(self.time_series)

    mp_distances = np.full(n - self.window_length + 1, np.inf)
    mp_indices = np.full(n - self.window_length + 1, -1)

    comparison_segments = np.array([
      time_series_np[j:j + self.window_length]
      for j in range(n - self.window_length + 1)
    ])

    for i in range(n - self.window_length + 1):
      query = comparison_segments[i]

      # Calculate distances for all segments at once using L2 norm across factors (broadcasting)
      distances = np.linalg.norm(comparison_segments[:, None, :] - query, axis=1)

      # Set the distance of the query itself to infinity to avoid self-comparison
      distances[i] = np.inf

      # Update minimum distance and index
      mp_distances[i] = np.min(distances)
      mp_indices[i] = np.argmin(distances)

    return mp_distances, mp_indices
  
  def _compute_with_cuda(self) -> None:
    """Compute matrix profile using CUDA (CuPy) with proper z-normalization."""
    time_series_np = self.time_series["values"].to_numpy().flatten()
    n = len(time_series_np)
    
    # Create sliding windows
    segments_np = np.lib.stride_tricks.sliding_window_view(time_series_np, self.window_length)
    segments_gpu = cp.array(segments_np, dtype=cp.float32)
    
    means = cp.mean(segments_gpu, axis=1)
    stds = cp.std(segments_gpu, axis=1)
    
    # Z-normalize all segments
    segments_normalized = ((segments_gpu - means[:, None]) / stds[:, None])
    
    mp_distances = cp.full(n - self.window_length + 1, cp.inf, dtype=cp.float32)
    mp_indices = cp.full(n - self.window_length + 1, -1, dtype=cp.int32)
    
    for i in range(n - self.window_length + 1):
      query = segments_normalized[i]
      
      # Compute z-normalized Euclidean distance
      dot_product = cp.sum(segments_normalized * query, axis=1)
      distances = cp.sqrt(2 * self.window_length * (1 - dot_product))
      
      # Exclude self-matches
      distances[i] = cp.inf
      
      # Update matrix profile
      mp_distances[i] = cp.min(distances)
      mp_indices[i] = cp.argmin(distances)
    
    return cp.asnumpy(mp_distances), cp.asnumpy(mp_indices)

  # Needs to be analyzed
  def _compute_with_cuda_bad(self) -> None:
    """Compute matrix profile using CUDA (CuPy)."""

    time_series_np = self.time_series["values"].to_numpy().flatten()
    n = len(time_series_np)

    comparison_segments_np = np.lib.stride_tricks.sliding_window_view(time_series_np, self.window_length)

    # Move comparison_segments_np from CPU to GPU
    comparison_segments_gpu = cp.array(comparison_segments_np, dtype=cp.float32)

    mp_distances = cp.full(n - self.window_length + 1, cp.inf, dtype=cp.float32)
    mp_indices = cp.full(n - self.window_length + 1, -1, dtype=cp.int32)
    
    for i in range(n - self.window_length + 1):
      query = comparison_segments_gpu[i]

      # Calculate distances for all segments at once using L2 norm across factors (broadcasting)
      distances = cp.linalg.norm(comparison_segments_gpu - query, axis=1)

      # Set the distance of the query itself to infinity to avoid self-comparison
      distances[i] = cp.inf

      # Update minimum distance and index
      mp_distances[i] = cp.min(distances)
      mp_indices[i] = cp.argmin(distances)

    return cp.asnumpy(mp_distances), cp.asnumpy(mp_indices)


  def compute(self, use_cuda: bool = False) -> None:
    """Compute matrix profile using CUDA (CuPy) or CPU, whichever is available."""

    if len(self.time_series) < self.window_length:
      log_table("Matrix Profile Computation", "ERROR", f"The time series length ({len(self.time_series)}) must be at least as large as the window length ({self.window_length}).", level="ERROR")
      return
    
    computation_type = "GPU" if use_cuda else "CPU"
    
    log_table("Matrix Profile Computation", "INFO", f"Starting matrix profile computation with {computation_type}")

    log_computation_info(use_cuda)

    if cp.cuda.runtime.getDeviceCount() == 0 and use_cuda:
      log_table("Matrix Profile Computation", "ERROR", f"Computing with CUDA failed. CUDA not available. Please ensure CUDA, CDNN, and System Variables are set.", level="ERROR")

    res = None

    if use_cuda:
      res = self._compute_with_cuda()
    
    else:
      res = self._compute_with_cpu()
    
    if len(res) == 0:
      log_table("Matrix Profile Computation", "ERROR", "Matrix profile computation returned an empty result.", level="ERROR")
      
    self.profile_distances, self.profile_indices = res

    log_table("Matrix Profile Computation", "INFO", "Matrix profile computation complete.")


  def find_motifs(self, num_motifs: int = 1, threshold: float = None, threshhold_percentile: float = 30.0) -> List[int]:
    """Find motifs in the matrix profile."""
    
    # Motif indices can be determined by finding indices with the smallest distances

    if not self.profile_distances.any():
      log_table("Matrix Profile Motif Detection", "ERROR", "Matrix profile distances are not computed. Run `compute()` first.", level="ERROR")
      return []
    
    if not threshold:
      threshold = np.percentile(self.profile_distances, threshhold_percentile)
  
    if threshold > np.max(self.profile_distances):
      log_table("Matrix Profile Motifs Detection", "Warning", "Motifs threshold is too low; no distances exceed this value.", level="WARNING")
      return []

    log_table("Matrix Profile Motif Detection", "INFO", f"Finding {num_motifs} motifs.")

    sorted_indices = np.argsort(self.profile_distances)
    motifs = sorted_indices[self.profile_distances[sorted_indices] <= threshold].tolist()

    if num_motifs:
      motifs = motifs[:num_motifs]  

    log_table("Matrix Profile Motif Detection", "INFO", f"Motifs found at indices: {motifs}")

    return motifs


  def find_discords(self, num_discords: int = None, threshold: float = None, threshhold_percentile: float = 85.0) -> List[int]:
    """Find discords in the matrix profile."""
    # Motif indices can be determined by finding indices with the largest distances

    if not self.profile_distances.any():
      log_table("Matrix Profile Discord Detection", "ERROR", "Matrix profile distances are not computed. Run `compute()` first.", level="ERROR")
      return []
    
    if not threshold:
      threshold = np.percentile(self.profile_distances, threshhold_percentile)

    high_discords = self.profile_distances > threshold

    if np.sum(high_discords) == 0:
        log_table("Matrix Profile Discord Detection", "WARNING", "No discords found above the specified threshold.", level="WARNING")
        return []
    
    log_table("Matrix Profile Discord Detection", "INFO", f"Finding {num_discords} discords.")

    sorted_indices = np.argsort(self.profile_distances)[::-1]
    discords = sorted_indices[:num_discords].tolist()

    if not num_discords:
      discords = discords[:num_discords]

    log_table("Discord Detection", "INFO", f"Discords found at indices: {discords}")

    return discords


  def plot_profiles(self, motif_threshold: float = None, discord_threshold: float = None) -> None:
    """Visualize the time series and matrix profile with optional thresholds."""

    log_table("Matrix Profile Plotting", "INFO", "Plotting matrix profile and time series.")

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(self.time_series, label='Time Series', color='blue')
    plt.title('Time Series')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(self.profile_distances, label='Matrix Profile Distances', color='orange')

    if motif_threshold is not None:
      plt.axhline(y=motif_threshold, color='green', linestyle='--', label='Motif Threshold')

    if discord_threshold is not None:
      plt.axhline(y=discord_threshold, color='red', linestyle='--', label='Discord Threshold')

    plt.title('Matrix Profile')
    plt.xlabel('Index')
    plt.ylabel('Distance')
    plt.legend()

    plt.tight_layout()
    plt.show()

    log_table("Matrix Profile Plotting", "Step: Plotting", "Plotting complete.")