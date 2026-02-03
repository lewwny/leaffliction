import json
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict, Any


# consts
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)

