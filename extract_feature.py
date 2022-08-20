import math
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_hog(img, img_name, cell_size=8, bin_size=9, img_save_dir=None):
    print("提取{}的方向梯度直方图特征...".format(img_name))
    height = len(img)
    width = len(img[0])

    hog_fea = np.zeros((bin_size,))

    dst_x = cv2.Sobel(img, cv2.CV_64FC1, 1, 0, ksize=5, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    dst_y = cv2.Sobel(img, cv2.CV_64FC1, 0, 1, ksize=5, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    gradient_magnitude = cv2.addWeighted(dst_x, 0.5, dst_y, 0.5, 0, dtype=-1)
    gradient_angle = cv2.phase(dst_x, dst_y, angleInDegrees=True)  # 梯度方向没有问题
    gradient_magnitude = abs(gradient_magnitude)  # 最终得到的梯度大小没有问题

    x_shape = height // cell_size
    y_shape = width // cell_size
    z_shape = bin_size

    cell_gradient_vector = np.zeros((x_shape, y_shape, z_shape + 1))
    cell_magnitude = np.zeros((cell_size, cell_size))
    cell_angle = np.zeros((cell_size, cell_size))

    for i in range(x_shape):
        for j in range(y_shape):
            for k in range(cell_size):
                for v in range(cell_size):
                    cell_magnitude[k][v] = gradient_magnitude[i * cell_size + k][j * cell_size + v]
                    cell_angle[k][v] = gradient_angle[i * cell_size + k][j * cell_size + v]

            for k in range(cell_size):
                for v in range(cell_size):
                    gradient_strength = float(cell_magnitude[k][v])
                    gradient_angle_tmp = int(cell_angle[k][v])
                    min_angle = int(gradient_angle_tmp // 40)
                    max_angle = int((min_angle + 1) % 40)
                    mod = float(gradient_angle_tmp % 40)
                    cell_gradient_vector[i][j][min_angle] += (gradient_strength * (1 - (mod / 40)))
                    cell_gradient_vector[i][j][max_angle] += (gradient_strength * (mod / 40))

    for k in range(z_shape):
        for i in range(x_shape):
            for j in range(y_shape):
                hog_fea[k] += cell_gradient_vector[i][j][k]

    # cv2.imshow("origin", img) # 原图
    # cv2.imshow("res", gradient_magnitude) # 梯度图
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 如果没有指定结果图保存路径，则不保存图片，只返回hog特征
    if img_save_dir is None:
        return hog_fea

    if not os.path.isdir(img_save_dir):
        os.mkdir(img_save_dir)
    img_hog_mag_save_path = os.path.join(img_save_dir, img_name + "_hog_magnitude.png")
    img_hog_dirt_save_path = os.path.join(img_save_dir, img_name + "_hog_direction.png")
    img_hog_hist_save_path = os.path.join(img_save_dir, img_name + "_hog_hist.png")
    cv2.imwrite(img_hog_mag_save_path, gradient_magnitude)
    cv2.imwrite(img_hog_dirt_save_path, gradient_angle)
    plot_hog(hog_fea, "HOG Feature Hist", img_hog_hist_save_path)
    return hog_fea, img_hog_mag_save_path, img_hog_dirt_save_path, img_hog_hist_save_path

def plot_hog(hog_fea, title, save_path):
    plt.figure(figsize=(6, 36 / 8))
    label_list = ['0', '20', '40', '60', '80', '100', '120', '140', '160']
    font = {'family': 'Times New Roman', 'size': 12}
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    plt.grid(linestyle="--", zorder=0)

    x = range(len(hog_fea))
    y = normalization(hog_fea)

    rects = plt.bar(x=x, height=y, width=0.6, alpha=0.8, color='steelblue', zorder=3)
    plt.ylim(0, 0.25)

    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%.2f' % height,
                 ha='center', va='bottom', fontdict=font)

    plt.xticks([index for index in x], label_list, fontproperties='Times New Roman', size=12)
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xlabel("gradient direction", fontsize=14, fontweight='bold', fontdict=font)
    plt.ylabel("gradient amplitude", fontsize=14, fontweight='bold', fontdict=font)
    plt.title(title)
    plt.savefig(save_path, format='png')
    # plt.show()

def normalization(nums):
    nums = np.array(nums)
    sum = nums.sum()
    return nums / sum


def extract_glcm(img, img_name, angle=0, gray_level=16, img_save_dir=None):
    print("提取{}的灰度共生矩阵特征...".format(img_name))
    glcm_feaNum = 4
    glcm_fea = np.zeros((glcm_feaNum,), dtype=float)

    vec_glcm = calGLCM(img, angle, gray_level)
    if vec_glcm is None:
        return glcm_fea
    energy, idmoment, contrast, entropy = getGLCMFeatures(vec_glcm, gray_level)
    glcm_fea[0] = energy * 10000
    glcm_fea[1] = idmoment * 10000
    glcm_fea[2] = contrast * 10000
    glcm_fea[3] = entropy * 10000

    # 如果没有指定结果图保存路径，则不保存图片，只返回hog特征
    if img_save_dir is None:
        return glcm_fea

    # 将 glcm 从 16x16 放大至 512x512
    matGlcmMatrix = np.zeros((512, 512), dtype=float)
    mag_power = 512 // gray_level
    for i in range(512):
        for j in range(512):
            matGlcmMatrix[i][j] = vec_glcm[i // mag_power][j // mag_power]

    if not os.path.isdir(img_save_dir):
        os.mkdir(img_save_dir)
    img_gray_save_path = os.path.join(img_save_dir, img_name + "_gray.png")
    img_glcm_matrix_save_path = os.path.join(img_save_dir, img_name + "_glcm_matrix.png")
    cv2.imwrite(img_gray_save_path, img)
    cv2.imwrite(img_glcm_matrix_save_path, matGlcmMatrix)
    return glcm_fea, img_gray_save_path, img_glcm_matrix_save_path


def calGLCM(img, angle, gray_level):
    height = len(img)
    width = len(img[0])
    img = np.asarray(img);
    maxGrayValue = img.max()
    maxGrayLevel = maxGrayValue + 1

    if maxGrayLevel > gray_level:
        img = img // gray_level

    if angle == 0:
        vec_glcm = getGLCMHorison(img, height, width, gray_level)
    elif angle == 90:
        vec_glcm = getGLCMVertial(img, height, width, gray_level)
    elif angle == 45:
        vec_glcm = getGLCM45(img, height, width, gray_level)
    elif angle == 135:
        vec_glcm = getGLCM135(img, height, width, gray_level)
    else:
        print("glcm direction error")
        return None
    return vec_glcm


def getGLCMHorison(img, height, width, gray_level):
    vec_glcm = np.zeros((gray_level, gray_level), dtype=int)
    for i in range(height):
        for j in range(width - 1):
            row = img[i][j]
            col = img[i][j + 1]
            vec_glcm[row][col] += 1
    return vec_glcm


def getGLCMVertial(img, height, width, gray_level):
    vec_glcm = np.zeros((gray_level, gray_level), dtype=int)
    for i in range(height - 1):
        for j in range(width):
            row = img[i][j]
            col = img[i + 1][j]
            vec_glcm[row][col] += 1
    return vec_glcm


def getGLCM45(img, height, width, gray_level):
    vec_glcm = np.zeros((gray_level, gray_level), dtype=int)
    for i in range(height - 1):
        for j in range(width - 1):
            row = img[i][j]
            col = img[i + 1][j + 1]
            vec_glcm[row][col] += 1
    return vec_glcm


def getGLCM135(img, height, width, gray_level):
    vec_glcm = np.zeros((gray_level, gray_level), dtype=int)
    for i in range(height - 1):
        for j in range(1, width):
            row = img[i][j]
            col = img[i + 1][j - 1]
            vec_glcm[row][col] += 1
    return vec_glcm


def getGLCMFeatures(vec_glcm, gray_level):
    total = vec_glcm.sum()
    vec_glcm = vec_glcm / total  # 归一化
    energy = 0
    idmoment = 0
    contrast = 0
    entropy = 0
    for i in range(gray_level):
        for j in range(gray_level):
            energy += vec_glcm[i][j] * vec_glcm[i][j]
            idmoment += vec_glcm[i][j] / (1 + (i - j) * (i - j))
            contrast += (i - j) * (i - j) * vec_glcm[i][j]
            if vec_glcm[i][j] > 0:
                entropy -= vec_glcm[i][j] * math.log(vec_glcm[i][j])
    return energy, idmoment, contrast, entropy


def extract_hog_and_glcm_feature(img_path, img_result_save_dir=None, hog_cell_size=8, hog_bin_size=9, glcm_gray_level=16, glcm_direction=0):
    src = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    # src = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(src, code=cv2.COLOR_BGR2GRAY, dstCn=0)

    img_full_name = img_path.split("/")[-1]
    img_name = os.path.splitext(img_full_name)[0]
    if img_result_save_dir is None:
        hog = extract_hog(img, img_name, img_save_dir=None)
        glcm = extract_glcm(img, img_name, img_save_dir=None)
        sem_fea = np.concatenate((hog, glcm), axis=0)
        return sem_fea
    else:
        hog, img_hog_mag_save_path, img_hog_dirt_save_path, img_hog_hist_save_path = extract_hog(img, img_name, img_save_dir=img_result_save_dir, cell_size=hog_cell_size, bin_size=hog_bin_size)
        glcm, img_gray_save_path, img_glcm_matrix_save_path = extract_glcm(img, img_name, img_save_dir=img_result_save_dir, gray_level=glcm_gray_level, angle=glcm_direction)
        sem_fea = np.concatenate((hog, glcm), axis=0)
        return sem_fea, img_hog_mag_save_path, img_hog_dirt_save_path, img_hog_hist_save_path, img_gray_save_path, img_glcm_matrix_save_path

if __name__ == '__main__':
    sem_fea, *_ = extract_hog_and_glcm_feature("test_code/中文路径/170C-24H-8000x.png", "./test_code/2021-10-28")
    print(sem_fea)

# print(img.shape)
# print(img)
# Correct Feature: 5497050	12646100	14027900	4348340	4727200	11317500	14319100	12501600	2178990	1771.7	8243.14	11811.7	21921.1
# hog, img_hog_mag_save_path, img_hog_dirt_save_path = extract_hog("./170C-24H-8000x.png",
#                                                                  img_save_dir="./images/test_hog")
# glcm, img_gray_save_path, img_glcm_matrix_save_path = extract_glcm("./170C-24H-8000x.png",
#                                                                    img_save_dir="./images/test_glcm")
# print(hog)
# print(glcm)
# sem_fea = np.concatenate((hog, glcm), axis=0)
# print(sem_fea)
