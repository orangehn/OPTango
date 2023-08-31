import os


def mkdir_of_file(path):
    os.makedirs(os.path.split(path)[0], exist_ok=True)


def get_save_path(save_root, bin_dir_suffix, bin_name):
    return os.path.join(save_root + bin_dir_suffix, bin_name + "_extract.pkl")  # unpair data


def get_ida_log_path(ida_log_dir, bin_dir_suffix, bin_name):
    return os.path.join(ida_log_dir + bin_dir_suffix, f'{bin_name}.log')
