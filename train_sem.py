import pickle

import numpy as np
from time import *

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

from data_load import LoadDataset, load_sem_data
from extract_feature import extract_hog_and_glcm_feature
from save_dir_generator import get_result_save_dir
import os
import shutil
from save_result import save_sem_fit_result
from util.common_function import *


def train_sem(data_dir, model_save_dir, data_type, hog_cell_size=8, hog_bin_size=9, glcm_gray_level=16, glcm_direction=0, degree=2, alpha=0.5):
    print()
    print("Training Info".center(150, "*"))
    begin_time = time()

    result_total_folder = "./results"
    result_sub_total_folder = os.path.join(result_total_folder, "sem")
    result_train_folder = os.path.join(result_sub_total_folder, "train")
    result_data_type_dir = os.path.join(result_train_folder, data_type)
    result_runs_folder = get_result_save_dir()  # 保存每次训练结果的子目录
    result_generate_dir = os.path.join(result_data_type_dir, result_runs_folder)  # 保存结果的完整目录
    result_model_dir = os.path.join(result_generate_dir, "models")  # 保存结果中模型的完整目录
    result_scaler_dir = os.path.join(result_model_dir, "scalers")  # 保存结果中模型的完整目录
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
    if not os.path.isdir(result_scaler_dir):
        os.mkdir(result_scaler_dir)

    data, labels = load_sem_data(data_dir, hog_cell_size=hog_cell_size, hog_bin_size=hog_bin_size,
                             glcm_gray_level=glcm_gray_level, glcm_direction=glcm_direction)

    common_name = "{}-Lasso-degree={}-alpha={}".format(data_type, degree, alpha)
    mm_scaler_name = "{}_minmiax_scaler.pickle".format(data_type)
    std_scaler_name = "{}_standard_scaler.pickle".format(data_type)
    model_name = common_name + ".pickle"

    # print(data)
    # print(labels)
    # 每次训练保存对应的scaler
    mm_scaler = MinMaxScaler()
    data = mm_scaler.fit_transform(data)

    mm_scaler_result_path = os.path.join(result_scaler_dir, mm_scaler_name)
    with open(mm_scaler_result_path, 'wb') as f:
        pickle.dump(mm_scaler, f)

    poly = PolynomialFeatures(degree=degree)
    data = poly.fit_transform(data)
    combinations = poly.powers_

    std_scaler = StandardScaler()
    data = std_scaler.fit_transform(data)

    std_scaler_result_path = os.path.join(result_scaler_dir, std_scaler_name)
    with open(std_scaler_result_path, 'wb') as f:
        pickle.dump(std_scaler, f)

    lasso = Lasso(alpha=alpha)
    lasso.fit(data, labels)
    predict_labels = lasso.predict(data)

    save_sem_fit_json(data_type, list(labels), list(predict_labels))

    # 生成一个保存的目录save_dir ,传入绘制图像的函数中
    train_result_pic_path = os.path.join(result_generate_dir, common_name + ".png")
    save_sem_fit_result(y_true=labels, y_predict=predict_labels, degree=degree, alpha=alpha, save_path=train_result_pic_path)

    model_result_path = os.path.join(result_model_dir, model_name)
    with open(model_result_path, 'wb') as f:
        pickle.dump(lasso, f)
    # print(predict_labels)

    train_mse = mean_squared_error(labels, predict_labels)

    end_time = time()
    run_time = end_time - begin_time
    print("训练完成!")
    print('训练时间为: ', run_time, "s")

    best_model_save_dir = model_save_dir
    best_scaler_save_dir = os.path.join(best_model_save_dir, "scalers")
    if not os.path.isdir(best_model_save_dir):
        os.mkdir(best_model_save_dir)
    if not os.path.isdir(best_scaler_save_dir):
        os.mkdir(best_scaler_save_dir)

    best_model_path_in_model_dir = os.path.join(best_model_save_dir, model_name)
    mm_scaler_path_in_model_dir = os.path.join(best_scaler_save_dir, mm_scaler_name)
    std_scaler_path_in_model_dir = os.path.join(best_scaler_save_dir, std_scaler_name)
    # if judge_is_copy_model_to_model_dir(sem_or_ftir="sem", data_type=data_type, cur_mse=train_mse,
    #                                     best_model_path_in_model_dir=best_model_path_in_model_dir):
    shutil.copy(model_result_path, best_model_save_dir)
    shutil.copy(mm_scaler_result_path, best_scaler_save_dir)
    shutil.copy(std_scaler_result_path, best_scaler_save_dir)

    print("*" * 150)
    print()
    return best_model_path_in_model_dir, mm_scaler_path_in_model_dir, std_scaler_path_in_model_dir, \
           train_result_pic_path, train_mse

if __name__ == "__main__":
    train_sem("./data/sem/溶剂型-200x/train", "./models", "溶剂型-200x",
              hog_cell_size=8, hog_bin_size=9, glcm_gray_level=16, glcm_direction=0, degree=3, alpha=0.5)




