from torch.utils.data import DataLoader
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.optim as optim
import cpuinfo
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

class PyTorchModel:
  def __init__(self, input_size, output_size, model_type='feedforward', hidden_layers=[64, 32], device=None):
    """
    Initialize the PyTorch model.
7
    Parameters:
        input_size (int): Number of input features.
        output_size (int): Number of output classes.
        model_type (str): Type of neural network ('feedforward', 'cnn', 'rnn').
        hidden_layers (list): List of integers representing hidden layer sizes for feedforward models.
        device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').
    """
  
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = self._build_model(input_size, output_size, model_type, hidden_layers).to(self.device)

    if torch.cuda.is_available():
        logger.info("Computing via CUDA selected.")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

        gpu_info = []
        for i in range(torch.cuda.device_count()):
            gpu_info.append([
                i,
                torch.cuda.get_device_name(i),
                torch.cuda.get_device_capability(i),
                torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)
            ])

        # Print GPU information as a table
        headers = ["Device ID", "Name", "Compute Capability", "Total Memory (MB)"]
        logger.info(tabulate(gpu_info, headers=headers, tablefmt="grid"))

    else:
        logger.info(f"Computing via CPU selected.")
        
        cpu_info = [
          [
            cpuinfo.get_cpu_info()['brand_raw'],
            torch.get_num_threads()
          ]
        ]

        headers = ["Processor Name", "Num of Available Threads"]
        logger.info(tabulate(cpu_info, headers=headers, tablefmt="grid"))

    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.Adam(self.mode.parameters())

  def _build_model(self, input_size, output_size, model_type, hidden_layers):
    if model_type == 'feedforward':
      layers = []
      last_size = input_size
      for hidden_size in hidden_layers:
        layers.append(nn.Linear(last_size, hidden_size))
        layers.append(nn.ReLU())
        last_size = hidden_size
      layers.append(nn.Linear(last_size, output_size))
      return nn.Sequential(*layers)

    elif model_type == 'rnn':
      return nn.Sequential(
        nn.LSTM(input_size, hidden_layers[0], batch_first=True),
        nn.Linear(hidden_layers[0], output_size)
      )

    else:
        raise ValueError("Invalid model type. Choose from 'feedforward' or 'rnn'.")


  def train(self, dataloader: DataLoader, num_epochs: int = 10):
    """Train the model."""
    self.model.train()
    for epoch in range(num_epochs):
      total_loss = 0

      for inputs, labels in dataloader:
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()  # Clear gradients
        outputs = self.model(inputs)  # Forward pass
        loss = self.criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass
        self.optimizer.step()  # Update weights
        total_loss += loss.item()

      print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')

    def evaluate(self, dataloader: DataLoader):
      """Evaluate the model."""
      self.model.eval()
      correct = 0
      total = 0

      with torch.no_grad():
        for inputs, labels in dataloader:
          inputs, labels = inputs.to(self.device), labels.to(self.device)
          outputs = self.model(inputs)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          
        accuracy = correct / total
        print(f'Accuracy: {accuracy:.4f}')
        return accuracy

  def save_model(self, file_path: str):
    """Save the model state to a file."""
    torch.save(self.model.state_dict(), file_path)

  def load_model(self, file_path: str):
    """Load the model state from a file."""
    self.model.load_state_dict(torch.load(file_path))
    self.model.to(self.device)