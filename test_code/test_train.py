from data_load import LoadDataset
from extract_feature import extract_hog_and_glcm_feature
import numpy as np

dataset = LoadDataset("../data/sem/水溶性-200x/train")
print(len(dataset.data_info))

data = []
labels = []
for item in dataset.data_info:
    img_label = item[1]
    if img_label < 0 or img_label > 50:
        continue
    img_path = item[0]
    img_features = extract_hog_and_glcm_feature(img_path)
    data.append(img_features)
    labels.append(img_label)
print(np.array(data).shape)
print(np.array(labels))
