import argparse
import os
import shutil
from tqdm import tqdm


def merge(*dirs, save_dir="merge", mode='move'):
    assert mode in ['move', 'copy']
    func = shutil.move if mode == 'move' else shutil.copy
    for _dir in dirs:
        if os.path.isdir(_dir):
            for root, sub_dirs, files in tqdm(os.walk(_dir)):
                for file in files:
                    file = os.path.join(root, file)
                    rel_path = os.path.relpath(file, _dir)
                    save_path = os.path.join(save_dir, rel_path)
                    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
                    func(file, save_path)
    print(f"merge {dirs} into {save_dir}")


def main():
    """
    Usage:
        python  ../../../../../../package/simple_utils/shutil_expand.py merge \
        '"bsc_feat_train/opt_rm_jtrans/00000452", "bsc_feat_eval/opt_rm_jtrans/00000452", "bsc_feat_seval/opt_rm_jtrans/00000452"' \
        '{"save_dir": "merge/00000452", "mode": "copy"}'
        python  ../../../../../../package/simple_utils/shutil_expand.py merge \
        '"bsc_feat_train/opt_rm_jtrans/00000452_const", "bsc_feat_eval/opt_rm_jtrans/00000452_const", "bsc_feat_seval/opt_rm_jtrans/00000452_const"' \
        '{"save_dir": "merge/00000452_const", "mode": "copy"}'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("f", help="function to run")
    parser.add_argument("args", help="")
    parser.add_argument("kwargs", help="")
    args = parser.parse_args()
    eval(args.f)(*eval(args.args), **eval(args.kwargs))


if __name__ == "__main__":
    main()
