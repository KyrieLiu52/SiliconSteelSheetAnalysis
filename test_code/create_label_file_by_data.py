import os


def create_label_txt_by_data(data_dir = "../data/ftir/溶剂型/test/data", label_dir = "../data/ftir/溶剂型/test/labels"):
    data_file_list = os.listdir(data_dir)

    for data_file_name in data_file_list:
        print("data file name: ", data_file_name)
        pure_name = data_file_name.split(".")[0]
        year = input("data's year: ")
        label_file_name = pure_name + ".txt"
        with open(os.path.join(label_dir, label_file_name), 'w', encoding="utf-8") as file:
            file.write(year)
        file.close()