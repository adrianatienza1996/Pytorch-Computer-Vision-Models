import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import os
import cv2

voc_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

labels_to_indx = {voc_labels[i] : i for i in range(len(voc_labels))}
indx_to_labels = {i : voc_labels[i] for i in range(len(voc_labels))}

annotations_path = "F:/VOC Dataset/VOCdevkit/VOC2007/Annotations"
images_dir = "F:/VOC Dataset/VOCdevkit/VOC2007/JPEGImages"
# annotations_path = "F:/VOC Dataset/VOCdevkit/VOC2012/Annotations"
# images_dir = "F:/VOC Dataset/VOCdevkit/VOC2012/JPEGImages"

boxes = []
labels = []
images_id = []
shape = []

for xml_file in os.listdir(annotations_path):
    file_path = os.path.join(annotations_path, xml_file)
    tree = ET.parse(file_path)
    root = tree.getroot()
    for object in root.iter('object'):

        label = object.find('name').text.lower().strip()
        if label not in voc_labels:
            continue

        bbox = object.find('bndbox')
        try:
            xmin = np.int32(bbox.find('xmin').text) - 1
            ymin = np.int32(bbox.find('ymin').text) - 1
            xmax = np.int32(bbox.find('xmax').text) - 1
            ymax = np.int32(bbox.find('ymax').text) - 1

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(labels_to_indx[label])
            images_id.append(xml_file[:-4])

            image_path = os.path.join(images_dir, xml_file[:-4] + str(".jpg"))
            tmp_image = cv2.imread(image_path)
            shape.append(tmp_image.shape[:-1])

        except:
            print(xml_file)

dataset = np.zeros((len(labels), 8), dtype = np.int32)
for i in range (len(labels)):
    dataset[i, 0] = images_id[i]
    dataset[i, 1:3] = shape[i]
    dataset[i, 3:7] = boxes[i]
    dataset[i, -1] = labels[i]

dataset = pd.DataFrame(dataset, columns=["image_id", "height", "width", "x_min", "y_min", "x_max", "y_max", "label"])
dataset.to_csv("test_dataset.csv", index=False)
print("CSV Created and saved")
