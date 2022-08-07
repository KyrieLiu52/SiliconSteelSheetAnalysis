import pickle

import numpy as np
from time import *

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

from data_load import LoadDataset, load_sem_data, load_ftir_data
from extract_feature import extract_hog_and_glcm_feature
from save_dir_generator import get_result_save_dir
import os
import shutil
from save_result import save_sem_fit_result, save_ftir_fit_result
from util.common_function import *


def train_ftir(data_dir, model_save_dir,data_type, alpha=0.5, l1_ratio=0.5):
    print()
    print("Training Info".center(150, "*"))
    begin_time = time()

    result_total_folder = "./results"
    result_sub_total_folder = os.path.join(result_total_folder, "ftir")
    result_train_folder = os.path.join(result_sub_total_folder, "train")
    result_data_type_dir = os.path.join(result_train_folder, data_type)
    result_runs_folder = get_result_save_dir()  # 保存每次训练结果的子目录
    result_generate_dir = os.path.join(result_data_type_dir, result_runs_folder)  # 保存结果的完整目录
    result_model_dir = os.path.join(result_generate_dir, "models")  # 保存结果中模型的完整目录
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
    if not os.path.isdir(result_model_dir):
        os.mkdir(result_model_dir)

    data, labels = load_ftir_data(data_dir)

    common_name = "{}-ElasticNet-alpha={}-l1ratio={}".format(data_type, alpha, l1_ratio)
    model_name = common_name + ".pickle"

    # print(data)
    # print(labels)

    enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    enet.fit(data, labels)

    predict_labels = enet.predict(data)

    save_ftir_fit_json(data_type, list(labels), list(predict_labels))

    # 生成一个保存的目录save_dir ,传入绘制图像的函数中
    train_result_pic_path = os.path.join(result_generate_dir, common_name + ".png")
    save_ftir_fit_result(y_true=labels, y_predict=predict_labels,
                         alpha=alpha, l1_ratio=l1_ratio, save_path=train_result_pic_path)

    model_result_path = os.path.join(result_model_dir, model_name)
    with open(model_result_path, 'wb') as f:
        pickle.dump(enet, f)
    # print(predict_labels)

    train_mse = mean_squared_error(labels, predict_labels)

    end_time = time()
    run_time = end_time - begin_time
    print("训练完成!")
    print('训练时间为: ', run_time, "s")

    best_model_save_dir = model_save_dir
    if not os.path.isdir(best_model_save_dir):
        os.mkdir(best_model_save_dir)

    best_model_path_in_model_dir = os.path.join(best_model_save_dir, model_name)
    if judge_is_copy_model_to_model_dir(sem_or_ftir="ftir", data_type=data_type, cur_mse=train_mse,
                                        best_model_path_in_model_dir=best_model_path_in_model_dir):
        shutil.copy(model_result_path, best_model_save_dir)

    print("*" * 150)
    print()
    return best_model_path_in_model_dir, train_result_pic_path, train_mse

if __name__ == "__main__":
    train_ftir("./data/ftir/溶剂型/train", "./models2", "溶剂型",
              alpha=0.16, l1_ratio=0.5)




