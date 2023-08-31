import os
import pickle
import shutil
import subprocess
import multiprocessing
import time
from util.pairdata import pairdata
from util.path_utils import mkdir_of_file, get_save_path, get_ida_log_path
import argparse


def write_path_to_path_config_script():
    cur_dir = os.path.split(__file__)[0]
    f = open(os.path.join(cur_dir, "path_config.py"), 'w')
    f.write("# this file is auto-generate by run.py, modify run.py if you need change the path in this file.\n")
    for path_name in ["data_path", 'strip_path', 'SAVE_ROOT']:
        path = os.path.abspath(eval(path_name)).replace('\\', '\\\\')
        f.write(f"{path_name} = '{path}'\n")
    f.close()


class BOBUtil(object):
    _WALK_RETURN = {
        "bob": lambda data_dir, bench, opt, binary: (bench, opt, binary),
        "path": lambda data_dir, bench, opt, binary: os.path.join(data_dir, bench, opt, binary),
        "path+bob": lambda data_dir, bench, opt, binary:
            (os.path.join(data_dir, bench, opt, binary), bench, opt, binary),
    }

    @staticmethod
    def walk_bob_dir(data_dir, bench_list=None, opt_list=None, return_mode='bob'):
        return_func = BOBUtil._WALK_RETURN[return_mode]
        for bench in os.listdir(data_dir):
            if bench_list is not None and bench not in bench_list:
                continue
            bench_dir = os.path.join(data_dir, bench)
            if os.path.isfile(bench_dir):
                continue

            for opt in os.listdir(bench_dir):
                if opt_list is not None and opt not in opt_list:
                    continue
                opt_dir = os.path.join(bench_dir, opt)
                if os.path.isfile(opt_dir):
                    continue

                for binary in os.listdir(opt_dir):
                    yield return_func(data_dir, bench, opt, binary)


def normalize_path(path):
    abs_path = os.path.abspath(path)
    return os.path.relpath(abs_path, os.getcwd())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="./dataset/example")
    parser.add_argument("--data-name", default="example")
    parser.add_argument("--ida-path", default="ida/idat64.exe")
    parser.add_argument("--cpu-rate", type=float, default=0.5)
    args = parser.parse_args()

    ida_path = args.ida_path
    data_path, data_name = args.data_path, args.data_name
    strip_path = f"dataset_strip/{data_name}"
    script_path = "process.py"
    SAVE_ROOT = f"extract/{data_name}"  # dir of pickle files saved by IDA
    ida_log_dir = f"log/{data_name}"
    ida_idb_dir = f"idb/{data_name}"

    data_path = normalize_path(data_path)
    write_path_to_path_config_script()

    start = time.time()
    # bin_pathes = sorted([bin_path for bin_path in BOBUtil.walk_bob_dir(data_path, return_mode='path')])
    bin_pathes = sorted([os.path.join(root, file) for root, dirs, files in os.walk(data_path) for file in files])

    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count() * args.cpu_rate))
    for bin_path in bin_pathes:
        bin_dir, bin_name = os.path.split(bin_path)
        assert bin_dir.startswith(data_path), (bin_dir, data_path)
        bin_dir_suffix = bin_dir[len(data_path):]
        strip_bin_path = os.path.join(strip_path + bin_dir_suffix, f"{bin_name}.strip")
        if not os.path.exists(strip_bin_path):
            mkdir_of_file(strip_bin_path)
            cmd = f"strip -s {bin_path} -o {strip_bin_path}"
            print("[jTrans.run]", cmd)
            os.system(cmd)
            if not os.path.exists(strip_bin_path):  # if not x86 elf
                os.makedirs(os.path.split(strip_bin_path)[0], exist_ok=True)
                shutil.copy(bin_path, strip_bin_path)

        save_path = get_save_path(SAVE_ROOT, bin_dir_suffix, bin_name)
        if os.path.exists(save_path):
            try:
                res = pickle.load(open(save_path, 'rb'))
                continue
            except EOFError as e:  # wrong pkl file
                pass
        print("[jTrans.run] to obtain", save_path)

        log_path = get_ida_log_path(ida_log_dir, bin_dir_suffix, bin_name)
        idb_path = os.path.join(ida_idb_dir + bin_dir_suffix, f'{bin_name}.idb')
        mkdir_of_file(log_path)
        mkdir_of_file(idb_path)
        cmd = [ida_path, f'-L{log_path}', '-c', '-A', f'-S{script_path}', f'-o{idb_path}', f'{strip_bin_path}']
        print(" ".join(cmd))
        pool.apply_async(subprocess.call, args=(cmd,))
    pool.close()
    pool.join()

    # check whether generate .pkl is exists
    for bin_path in bin_pathes:
        bin_dir, bin_name = os.path.split(bin_path)
        bin_dir_suffix = bin_dir[len(data_path):]
        save_path = get_save_path(SAVE_ROOT, bin_dir_suffix, bin_name)
        if not os.path.exists(save_path):
            log_path = get_ida_log_path(ida_log_dir, bin_dir_suffix, bin_name)
            print(f"[jTrans.run] XXXXXXXXXXXXXXX fail extract ida feat of {bin_path} into {save_path},\n"
                  f"see log {log_path} for details.")

    print('[*] Features Extracting Done')
    # pairdata(SAVE_ROOT)
    end = time.time()
    print(f"[*] Time Cost: {end - start} seconds")
