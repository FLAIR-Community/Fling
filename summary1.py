import os
from tensorboard.backend.event_processing import event_accumulator


def summarize_tb(path):
    # Read a exp path and get the best acc.
    # Get the path of tb file that starts with "events.out.tfevents"
    for tmp_path in os.listdir(path):
        if str(tmp_path).startswith('events.out.tfevents'):
            tb_path = os.path.join(path, tmp_path)
            break
    # Read the tb file
    ea = event_accumulator.EventAccumulator(tb_path)
    ea.Reload()
    keys = [
        'after_aggregation_test/test_acc',
        'before_aggregation_test/test_acc',
        'finetune/test_acc'
    ]

    # Get the best acc in the keys above.
    res_dict = {}
    for key in keys:
        if key in ea.scalars.Keys():
            vals = ea.scalars.Items(key)
            vals = [k.value for k in vals]
            res_dict[key] = max(vals)

    return res_dict

def is_leaf_dir(path):
    # If it is a leaf dir, it must contain the tb file starts with "events.out.tfevents"
    # If it is not a dir, of course it is not a leaf dir. Return False.
    if not os.path.isdir(path):
        return False

    # Detect whether there is a tb file.
    for file in os.listdir(path):
        if str(file).startswith('events.out.tfevents'):
            return True
    return False


def list_exp_dirs(path):
    # Recursively get the exp directories.
    # If the path is a leaf dir, this means that it is exactly the exp dir, return itself.
    if is_leaf_dir(path):
        return [path]

    # Otherwise, recursively detect the leaf dirs.
    ret = []
    for dire in os.listdir(path):
        ret += list_exp_dirs(os.path.join(path, dire))
    return ret


def summarize_exp_dir(path):
    # Given a path, print the exp's acc.
    res = summarize_tb(path)
    print('######################################')
    print(f'exp_dir: {path}')
    print(f'max acc: {res}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True)
    args = parser.parse_args()

    exp_dirs = list_exp_dirs(args.root_path)
    for p in exp_dirs:
        summarize_exp_dir(p)
