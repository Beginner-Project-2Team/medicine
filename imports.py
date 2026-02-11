import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import json
import sys
import glob
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2.functional as F
import matplotlib.pyplot as plt
import matplotlib
import random
import cv2
# import albumentations as A
import math

# from albumentations.pytorch import ToTensorV2
from torchvision import tv_tensors
from torch.utils.data import Dataset, DataLoader
from torchvision.tv_tensors import Image as TVImage, BoundingBoxes
from PIL import Image, ImageFont, ImageDraw
from collections import defaultdict
from sklearn.model_selection import train_test_split

from collections import Counter
from tqdm import tqdm
from pprint import pprint
import shutil
import yaml
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)