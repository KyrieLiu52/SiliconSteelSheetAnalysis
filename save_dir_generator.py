import time

def get_result_save_dir():
    localTime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_dir = "result_" + localTime
    return save_dir

if __name__ == '__main__':
    get_result_save_dir()