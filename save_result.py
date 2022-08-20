import matplotlib.pyplot as plt
import os


def save_sem_fit_result(y_true, y_predict, degree, alpha, save_path):
    plt.figure(figsize=(6, 5))
    plt.suptitle("Lasso Fit Result\ndegree:{}, alpha:{}".format(degree, alpha))
    plt.plot(y_true, label="true")
    plt.plot(y_predict, label="predict")
    plt.legend()
    plt.savefig(save_path, format="png")
    # plt.show()

def save_ftir_fit_result(y_true, y_predict, alpha, l1_ratio, save_path):
    plt.figure(figsize=(6, 5))
    plt.suptitle("ElasticNet Fit Result\nalpha:{}, l1_ratio:{}".format(alpha, l1_ratio))
    plt.plot(y_true, label="true")
    plt.plot(y_predict, label="predict")
    plt.legend()
    plt.savefig(save_path, format="png")
    # plt.show()

#
def plot_infrared(data, file_name, save_dir, non_zero_coef):
    nums = len(data)
    x = range(2650, 2650 - nums, -1)

    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

    ftir_visualize_path = os.path.join(save_dir, file_name+"_all_ftir.png")
    ftir_section_path_1 = os.path.join(save_dir, file_name+"_section_1.png")
    ftir_section_path_2 = os.path.join(save_dir, file_name+"_section_2.png")
    ftir_section_path_3 = os.path.join(save_dir, file_name+"_section_3.png")
    ftir_section_path_4 = os.path.join(save_dir, file_name+"_section_4.png")
    ftir_coef_scatter = os.path.join(save_dir, file_name+"_scatter.png")

    plt.figure(figsize=(6, 36/8))
    plt.grid(linestyle="--", zorder=0)
    plt.ylim(75, 110)
    plt.plot(x, data, label=file_name)
    plt.legend()
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xlabel("WaveNumber", fontsize=14)
    plt.ylabel("Absorbance", fontsize=14)
    plt.savefig(ftir_visualize_path, format="png")

    span = nums // 4
    x = range(2650, 2650 - span, -1)
    index = range(0, 0 + span, 1)
    data_1 = data[index]
    plt.figure(figsize=(6, 36 / 8))
    plt.grid(linestyle="--", zorder=0)
    plt.ylim(75, 110)
    plt.plot(x, data_1, label="section_1")
    plt.legend()
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xlabel("WaveNumber", fontsize=14)
    plt.ylabel("Absorbance", fontsize=14)
    plt.savefig(ftir_section_path_1, format="png")

    x = range(2650 - span, 2650 - span*2, -1)
    index = range(0 + span, 0 + span*2, 1)
    data_1 = data[index]
    plt.figure(figsize=(6, 36 / 8))
    plt.grid(linestyle="--", zorder=0)
    plt.ylim(75, 110)
    plt.plot(x, data_1, label="section_2")
    plt.legend()
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xlabel("WaveNumber", fontsize=14)
    plt.ylabel("Absorbance", fontsize=14)
    plt.savefig(ftir_section_path_2, format="png")

    x = range(2650 - span*2, 2650 - span*3, -1)
    index = range(0 + span*2, 0 + span*3, 1)
    data_1 = data[index]
    plt.figure(figsize=(6, 36 / 8))
    plt.grid(linestyle="--", zorder=0)
    plt.ylim(75, 110)
    plt.plot(x, data_1, label="section_3")
    plt.legend()
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xlabel("WaveNumber", fontsize=14)
    plt.ylabel("Absorbance", fontsize=14)
    plt.savefig(ftir_section_path_3, format="png")

    x = range(2650 - span*3, 2650 - span*4, -1)
    index = range(0+span*3, 0 + span*4, 1)
    data_1 = data[index]
    plt.figure(figsize=(6, 36 / 8))
    plt.grid(linestyle="--", zorder=0)
    plt.ylim(75, 110)
    plt.plot(x, data_1, label="section_4")
    plt.legend()
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xlabel("WaveNumber", fontsize=14)
    plt.ylabel("Absorbance", fontsize=14)
    plt.savefig(ftir_section_path_4, format="png")

    plt.figure(figsize=(6, 36 / 8))
    x = range(len(non_zero_coef))
    plt.scatter(x, non_zero_coef)
    plt.savefig(ftir_coef_scatter, format="png")
    # plt.show()

    return ftir_visualize_path, ftir_section_path_1, ftir_section_path_2, ftir_section_path_3, ftir_section_path_4, ftir_coef_scatter


if __name__ == '__main__':
    # save_sem_fit_result([1,2,3,4,5],[5,4,3,2,1],3,0.5,"./test_code/test.png")
    plot_infrared()