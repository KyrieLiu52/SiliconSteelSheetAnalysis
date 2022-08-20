import os
import pickle

import matplotlib
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from data_load import load_one_sem_data_and_feature_pic, load_one_ftir_data
from save_dir_generator import get_result_save_dir
from save_result import plot_infrared
from util.common_function import load_sem_predict_param

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def predict_sem(img_path, model_path, save_folder, hog_cell_size, hog_bin_size, glcm_gray_level, glcm_direction):
    result_sem_dir = os.path.join(save_folder, "sem")
    predict_result_save_dir = os.path.join(result_sem_dir, get_result_save_dir())
    if not os.path.isdir(result_sem_dir):
        os.mkdir(result_sem_dir)
    if not os.path.isdir(predict_result_save_dir):
        os.mkdir(predict_result_save_dir)
    data, hog_mag_path, hog_dirt_path, hog_hist_path, gray_img_path, glcm_matrix_path = \
        load_one_sem_data_and_feature_pic(img_path=img_path, pic_save_dir=predict_result_save_dir,
                                          hog_cell_size=hog_cell_size, hog_bin_size=hog_bin_size,
                                          glcm_gray_level=glcm_gray_level, glcm_direction=glcm_direction)
    sem_feature = data
    if data.ndim == 1:
        data = data.reshape(1, -1)

    model_name = os.path.basename(model_path)
    degree, minmax_scaler_path, standard_scaler_path = load_sem_predict_param(model_name=model_name)
    if degree is None:
        return [0], "", "", "", "", "", ""

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
    predict_label = model.predict(data)

    if predict_label < 0:
        predict_label = [0]
    elif predict_label > 50:
        predict_label = [50]
    return predict_label, sem_feature, hog_mag_path, hog_dirt_path, hog_hist_path, gray_img_path, glcm_matrix_path



def predict_ftir(file_path, model_path, save_folder):
    result_ftir_dir = os.path.join(save_folder, "ftir")
    predict_result_save_dir = os.path.join(result_ftir_dir, get_result_save_dir())
    if not os.path.isdir(result_ftir_dir):
        os.mkdir(result_ftir_dir)
    if not os.path.isdir(predict_result_save_dir):
        os.mkdir(predict_result_save_dir)
    data = load_one_ftir_data(file_path)
    feature = data
    if data.ndim == 1:
        feature = data.reshape(1, -1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    predict_label = model.predict(feature)

    np_coef = np.array(model.coef_)
    non_zero_index = np.flatnonzero(np_coef)
    non_zero_coef = np_coef[non_zero_index]

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    ftir_visualize_path, ftir_section_path_1, ftir_section_path_2, ftir_section_path_3, ftir_section_path_4, ftir_coef_scatter = \
        plot_infrared(data, file_name, predict_result_save_dir, non_zero_coef)

    print(predict_label, "*****************************************")
    if predict_label < 0:
        predict_label = [0]
    elif predict_label > 50:
        predict_label = [50]
    return predict_label, ftir_visualize_path, ftir_section_path_1, ftir_section_path_2, ftir_section_path_3, ftir_section_path_4, ftir_coef_scatter


if __name__ == '__main__':
    # predict_label, hog_mag_path, hog_dirt_path, gray_path, glcm_matrix_path=\
    #     predict_sem(img_path="./data/sem/溶剂型-200x/train/data/90C-24H-200x.png",
    #             model_path="./results/sem/train/溶剂型-200x/result_20211030_002248/models/溶剂型-200x-Lasso-degree=2-alpha=0.5.pickle",
    #             img_type="溶剂型-200x",
    #             save_folder="./inference_result",
    #             hog_cell_size=8, hog_bin_size=9, glcm_gray_level=16, glcm_direction=0)
    # print(predict_label, hog_mag_path, hog_dirt_path, gray_path, glcm_matrix_path)
    p = predict_ftir("G:/小组较大/2021_09_12 东方电机 硅钢片 SEM 红外/1_数据/FTIR数据/红外分离数据/有机/190C-1D.csv", "./results/ftir/train/溶剂型/result_20211029_211158/models/溶剂型-ElasticNet-alpha=0.16-l1ratio=0.5.pickle", "./inference_result")
    print(p)
    # save_result_pic([0,10,13],[0.9,0.0,0.1],"")

# def predict_image(img_path, model_path, img_type, save_folder):
#     class_names = ["0", "10", "13"]
#     result_img_type_dir = os.path.join(save_folder, img_type)
#     predict_result_save_dir = os.path.join(result_img_type_dir, get_result_save_dir())
#     if not os.path.isdir(result_img_type_dir):
#         os.mkdir(result_img_type_dir)
#     if not os.path.isdir(predict_result_save_dir):
#         os.mkdir(predict_result_save_dir)
#
#     img_name = os.path.basename(img_path)
#     result_path = os.path.join(predict_result_save_dir, "predict_result_pic.png")   # 预测结果图的路径，用于保存结果和返回路径给前端
#     feature1_path = os.path.join(predict_result_save_dir, "mid_feature1_pic.png")   # 中间特征_1 的路径
#     feature2_path = os.path.join(predict_result_save_dir, "mid_feature2_pic.png")   # 中间特征_2 的路径
#     src_img_path = os.path.join(predict_result_save_dir, img_name)   # 将原图复制一份，保存在测试结果目录下
#
#     model = tf.keras.models.load_model(model_path)
#
#     img_init = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)  # 打开图片
#     # img_init = cv2.imread(img_path)
#     img_init = cv2.resize(img_init, (224, 224))
#     img = np.asarray(img_init)
#
#     model_slice_1 = tf.keras.models.Model(inputs=model.get_layer("mobilenetv2_1.00_224").input,
#                                           outputs=model.get_layer("mobilenetv2_1.00_224").get_layer("Conv1").output,
#                                           name="model_slice")
#     mid_model_1 = tf.keras.Sequential([
#         model.get_layer("rescaling"),
#         model_slice_1
#     ])
#     model_slice_feature_1 = mid_model_1.predict(img.reshape(1, 224, 224, 3))
#
#     model_slice_2 = tf.keras.models.Model(inputs=model.get_layer("mobilenetv2_1.00_224").input,
#                                           outputs=model.get_layer("mobilenetv2_1.00_224").get_layer(
#                                               "block_1_project_BN").output,
#                                           name="model_slice")
#     mid_model_2 = tf.keras.Sequential([
#         model.get_layer("rescaling"),
#         model_slice_2
#     ])
#     model_slice_feature_2 = mid_model_2.predict(img.reshape(1, 224, 224, 3))
#
#     visualize_feature_map(model_slice_feature_1, save_path=feature1_path)
#     visualize_feature_map(model_slice_feature_2, save_path=feature2_path)
#
#     outputs = model.predict(img.reshape(1, 224, 224, 3))
#     predict_index = np.argmax(outputs)
#     predict_label = class_names[predict_index]
#
#     predict_result = predict_label + "年"
#     confidence = outputs[0][predict_index]
#     save_result_pic(labels=[0, 10, 13], confidences=outputs[0], save_path=result_path)
#
#     shutil.copy(img_path, src_img_path)
#
#     # 返回预测结果，置信度，结果图的路径，中间特征层1的图片，中间特征层2的图片， 被检测的原图
#     return predict_result, confidence, result_path, feature1_path, feature2_path, src_img_path


