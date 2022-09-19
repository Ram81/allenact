from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import torch

for gpu_device in range(torch.cuda.device_count()):
  print(f"Attempting to start controlle on GPU {gpu_device}.")
  c = Controller(platform=CloudRendering, gpu_device=gpu_device)
  print(f"Started on GPU {gpu_device} successfully.")
  c.stop()