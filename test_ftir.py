import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from data_load import load_sem_data, load_ftir_data
from save_dir_generator import get_result_save_dir
import os

from save_result import save_sem_fit_result, save_ftir_fit_result

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


def test_ftir(test_data_dir, model_path, data_type, alpha=0.5, l1_ratio=0.5):
    print()
    print("Test Model Info".center(150, "*"))
    print("正在对测试集数据进行测试......")

    result_total_folder = "./results"
    result_sub_total_folder = os.path.join(result_total_folder, "ftir")
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

    data, labels = load_ftir_data(test_data_dir)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    predict_labels = model.predict(data)

    test_result_pic_path = os.path.join(result_generate_dir, "test_fit_result.png")
    save_ftir_fit_result(y_true=labels, y_predict=predict_labels, alpha=alpha, l1_ratio=l1_ratio,
                        save_path=test_result_pic_path)

    test_mse = mean_squared_error(labels, predict_labels)

    print("测试完成!")

    print("*" * 150)
    print()

    return test_result_pic_path, test_mse

if __name__ == '__main__':
    test_ftir("./data/ftir/溶剂型/test", "./models4/溶剂型-ElasticNet-alpha=0.16-l1ratio=0.5.pickle",
              "溶剂型", alpha=0.5, l1_ratio=0.16)


