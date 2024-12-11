from rich.console import Console
from rich.table import Table
import cpuinfo
import torch

misty_rose = "#FDE8E9"
fairy_tail = "#E3BAC6"
lilac = "#BC9EC1"
raisin_black = "#1F2232"

# Initialize console for Rich logging
console = Console()

def log_table(task: str, log_type: str, message: str, level: str = "STATUS", headers: list = None):
  """Utility function to log messages in table format with Rich."""
  
  # Create a table with the task at the top
  table = Table(title=f"Task: {task}", title_justify="center")

  # Add columns for step and status
  table.add_column("Log Type", justify="left")
  table.add_column("Status", justify="left")

  # Define color based on log level
  color = {
      "STATUS": "white",
      "WARNING": "yellow",
      "ERROR": "red",
      "SUCCESS": "green"
  }.get(level, "white")


  # Add row to the table with step and message
  table.add_row(log_type, message)

  # Print the colored table using Rich
  console.print(table, style=color)

  print("\n")
  print("\n")


def log_computation_info(use_cuda: True):
  """Log GPU and CPU information using custom logging functions."""
  
  table = None
  
  if use_cuda:
    table = Table(title=f"CUDA Device Info", title_justify="center")

    table.add_column("Device ID", justify="left")
    table.add_column("Name", justify="left")
    table.add_column("Compute Capability", justify="left")
    table.add_column("Total Memory (MB)", justify="left")

    for i in range(torch.cuda.device_count()):
      table.add_row(
        str(i),
        torch.cuda.get_device_name(i), 
        str(torch.cuda.get_device_capability(i)),
        str(torch.cuda.get_device_properties(i).total_memory / (1024 ** 2))  
      )
  
  else:
    table = Table(title=f"CPU Device Info", title_justify="center")

    table.add_column("Processor Name")
    table.add_column("Num of Available Threads")

    table.add_row(
      cpuinfo.get_cpu_info()['brand_raw'], 
      str(torch.get_num_threads())
    )
  
  console.print(table, style=misty_rose)

  print("\n")
  print("\n")