import os
import numpy as np
import pandas as pd

from extract_feature import extract_hog_and_glcm_feature

'''
    load image paths and labels
'''


class LoadDataset():
    def __init__(self, data_set_dir):
        self.data_info = self.get_data_and_label(data_set_dir)

    def get_data_and_label(self, data_set_dir):
        data_info = list()
        data_dir = os.path.join(data_set_dir, "data")
        labels_dir = os.path.join(data_set_dir, "labels")

        for item in os.listdir(data_dir):
            item_pure_name = os.path.splitext(item)[0]  # 获取不带扩展名的文件名 split extend
            label_file_name = item_pure_name + ".txt"
            label_file_path = os.path.join(labels_dir, label_file_name)
            with open(label_file_path, "r", encoding='utf-8') as f:
                content = f.read()
            if content is None:
                continue
            data_file_path = os.path.join(data_dir, item)
            data_info.append((data_file_path, float(content)))
        return data_info


def load_sem_data(data_dir, hog_cell_size=8, hog_bin_size=9, glcm_gray_level=16, glcm_direction=0):
    dataset = LoadDataset(data_dir)

    data = []
    labels = []
    for item in dataset.data_info:
        img_label = item[1]
        if img_label < 0 or img_label > 50:
            continue
        img_path = item[0]
        img_features = extract_hog_and_glcm_feature(img_path, hog_cell_size=hog_cell_size, hog_bin_size=hog_bin_size,
                                                    glcm_gray_level=glcm_gray_level, glcm_direction=glcm_direction)
        data.append(img_features)
        labels.append(img_label)
    return np.array(data), np.array(labels)


def load_ftir_data(data_dir):
    dataset = LoadDataset(data_dir)

    data = []
    labels = []
    for item in dataset.data_info:
        file_label = item[1]
        if file_label < 0 or file_label > 50:
            continue
        file_path = item[0]
        file_full_name = file_path.split("/")[-1]
        file_name = os.path.splitext(file_full_name)[0]
        features = pd.read_csv(file_path)["Absorbance"].to_numpy()
        print("提取{}的红外特征...".format(file_name))
        data.append(features)
        labels.append(file_label)
    return np.array(data), np.array(labels)


def load_one_sem_data_and_feature_pic(img_path, pic_save_dir, hog_cell_size=8, hog_bin_size=9, glcm_gray_level=16,
                                      glcm_direction=0):
    img_features, img_hog_mag_save_path, img_hog_dirt_save_path, img_hog_hist_save_path, img_gray_save_path, img_glcm_matrix_save_path \
        = extract_hog_and_glcm_feature(img_path=img_path, img_result_save_dir=pic_save_dir,
                                       hog_cell_size=hog_cell_size, hog_bin_size=hog_bin_size,
                                       glcm_gray_level=glcm_gray_level, glcm_direction=glcm_direction)
    return img_features, img_hog_mag_save_path, img_hog_dirt_save_path, img_hog_hist_save_path, img_gray_save_path, img_glcm_matrix_save_path

def load_one_ftir_data(file_path):
    data = pd.read_csv(file_path, header=None)
    wn_and_ab_threhold = 2000  # 判断该列是WaveNumber或Absorbance的指标
    min_wavenumber = 650
    max_wavenumber = 2650
    # 判断csv文件是否有表头
    if type(data[0][0]) == type("S"):
        data = pd.read_csv(file_path)
        # 判断两列是否是WaveNumber和Absorbance的顺序
        if data[data.columns[0:1]].to_numpy().max() < wn_and_ab_threhold:
            columns = data.columns
            # 交换两列
            data = data.loc[:, [columns[1], columns[0]]]
        columns = data.columns
        data = data[(data[columns[0]] >= min_wavenumber) & (data[columns[0]] <= max_wavenumber)]
    else:
        if data[data.columns[0:1]].to_numpy().max() < wn_and_ab_threhold:
            columns = data.columns
            # 交换两列
            data = data.loc[:, [columns[1], columns[0]]]
        columns = data.columns
        data = data[(data[columns[0]] >= min_wavenumber) & (data[columns[0]] <= max_wavenumber)]
        data.columns = ['Wavenumber', 'Absorbance']
    return data['Absorbance'].to_numpy()

if __name__ == '__main__':
    # dataset = LoadDataset("./data/ftir/水溶性/test")
    # print(dataset.data_info)
    data = load_one_ftir_data("./data/ftir/溶剂型/test/data/190C-150H.csv")
    print(data)

