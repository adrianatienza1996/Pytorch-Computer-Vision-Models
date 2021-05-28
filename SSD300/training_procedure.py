import pandas as pd
import numpy as np
from utils import *
from SSD_Dataset import *
from model import SSD_300


train_annotations_path = ""
test_annotations_path = ""

train_images_dir = ""
test_images_dir = ""

train_annotations = np.array(pd.read_csv(train_annotations_path))
test_annotations = np.array(pd.read_csv(test_annotations_path))

train_loader = get_loader(train_annotations, train_annotations_path)
test_loader = get_loader(test_annotations, test_annotations_path)


num_classes = None
model = SSD_300(num_classes=num_classes)

