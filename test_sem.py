import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from data_load import load_sem_data
from save_dir_generator import get_result_save_dir
import os

from save_result import save_sem_fit_result

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


def test_sem(test_data_dir, model_path, data_type, minmax_scaler_path, standard_scaler_path,
               hog_cell_size=8, hog_bin_size=9, glcm_gray_level=16, glcm_direction=0, degree=2, alpha=0.5):
    print()
    print("Test Model Info".center(150, "*"))
    print("正在对测试集数据进行测试......")

    result_total_folder = "./results"
    result_sub_total_folder = os.path.join(result_total_folder, "sem")
    result_train_folder = os.path.join(result_sub_total_folder, "test")
    result_data_type_dir = os.path.join(result_train_folder, data_type)
    result_runs_folder = get_result_save_dir()  # 保存每次训练结果的子目录
    result_generate_dir = os.path.join(result_data_type_dir, result_runs_folder)  # 保存结果的完整目录
    if not os.path.isdir(result_total_folder):
        os.mkdir(result_total_folder)
    if not os.path.isdir(result_sub_total_folder):
        os.mkdir(result_sub_total_folder)
    if not os.path.isdir(result_train_folder):
        os.mkdir(result_train_folder)
    if not os.path.isdir(result_data_type_dir):
        os.mkdir(result_data_type_dir)
    if not os.path.isdir(result_generate_dir):
        os.mkdir(result_generate_dir)

    data, labels = load_sem_data(test_data_dir, hog_cell_size=hog_cell_size, hog_bin_size=hog_bin_size,
                                 glcm_gray_level=glcm_gray_level, glcm_direction=glcm_direction)

    with open(minmax_scaler_path, "rb") as f:
        mm_scaler = pickle.load(f)
    data = mm_scaler.transform(data)
    poly = PolynomialFeatures(degree=degree)
    data = poly.fit_transform(data)
    with open(standard_scaler_path, "rb") as f:
        std_scaler = pickle.load(f)
    data = std_scaler.transform(data)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    predict_labels = model.predict(data)

    test_result_pic_path = os.path.join(result_generate_dir, "test_fit_result.png")
    save_sem_fit_result(y_true=labels, y_predict=predict_labels, degree=degree, alpha=alpha,
                        save_path=test_result_pic_path)

    test_mse = mean_squared_error(labels, predict_labels)

    print("测试完成!")

    print("*" * 150)
    print()

    return test_result_pic_path, test_mse

if __name__ == '__main__':
    test_sem("./data/sem/溶剂型-200x/test", "./models2/溶剂型-200x-Lasso-degree=3-alpha=0.5.pickle", "溶剂型-200x",
             "./models2/scalers/溶剂型-200x_minmiax_scaler.pickle", "./models2/scalers/溶剂型-200x_standard_scaler.pickle",
             hog_cell_size=8, hog_bin_size=9, glcm_gray_level=16, glcm_direction=0, degree=3, alpha=0.5)


