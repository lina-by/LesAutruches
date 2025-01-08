import torchvision.transforms as transforms
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt