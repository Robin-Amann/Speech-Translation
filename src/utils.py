import os
import random
from typing import Literal
from pathlib import Path
type Split = Literal["train", "dev", "test"]
from typing import Callable

def print_directory_tree(root_path: str, prefix="", depth=0):
    if depth == 0 :
        if root_path.endswith("/") :
            root_path = root_path[:-1]

        print(root_path.split("/")[-1])

    try:
        entries = sorted(
            [e for e in os.listdir(root_path)
             if os.path.isdir(os.path.join(root_path, e))]
        )
    except OSError as e:
        print(f"Error accessing '{root_path}': {e}")
        return

    for i, entry in enumerate(entries):
        connector = "└─ " if i == len(entries) - 1 else "├─ "
        print(prefix + connector + entry)
        next_prefix = prefix + ("   " if i == len(entries) - 1 else "│  ")
        print_directory_tree(os.path.join(root_path, entry), next_prefix, depth=depth+1)



def save_dataset(
        dataset: list[dict], 
        save_file: Callable[[Split, dict], None], 
        split_keys: list[str]  
) :
    if len(split_keys) == 0 :
        for entry in dataset :
            save_file(None, entry)
        return

    key, remaining_keys = split_keys[0], split_keys[1:]

    groups: dict[str, list[dict]] = dict()
    for entry in dataset :
        k = entry[key]
        if k not in groups :
            groups[k] = []
        groups[k].append(entry)
    
    for k, values in groups.items() :
        save_dataset(values, save_file, remaining_keys)


def load_dataset(target_dir: str | Path, load_file: Callable[str, tuple[Split, dict] | None], glob_regex="*") :
    "returns either a dataset or a train, dev, test split"
    target_dir = Path(target_dir)

    dataset, train, dev, test = [], [], [], []
    
    for path in target_dir.rglob(glob_regex):   # recursive
        if not path.is_file():
            continue

        result = load_file(path)
        if type(result) == None :
            continue

        split, entry = result
        if split == None :
            dataset.append(entry)
        elif split == "train" :
            train.append(entry)
        elif split == "dev" :
            dev.append(entry)
        else :
            test.append(entry)
    
    if len(dataset) == 0 :
        return train, dev, test
    else :
        return dataset


def split_and_save_dataset(
        dataset: list[dict], 
        save_file: Callable[[Split, dict], None], 
        split_keys: list[str], 
        train_dev_test: tuple[int, int, int]=(80, 10, 10), 
        _seed: int=None
) :
    if sum(train_dev_test) == 1 :
        train_dev_test = [ x * 100 for x in train_dev_test ]
    assert sum(train_dev_test) == 100

    if len(split_keys) == 0 :
        shuffled = dataset.copy()   # only shallow copy since only the order is changed
        if _seed != None :
            random.seed(_seed)
        random.shuffle(shuffled)
        n = len(shuffled)
        n1 = int(n * train_dev_test[0] / 100)
        n2 = int(n * (train_dev_test[0] + train_dev_test[1]) / 100)
        assert 0 < n1 < n2 < n, f"{dataset[0]["speaker_id"]}, {dataset[0]["cohort"]}"      # in each split it at least one entry
        train = shuffled[:n1]
        dev = shuffled[n1:n2]
        test = shuffled[n2:]
        for split, name in [(train, "train"), (dev, "dev"), (test, "test")] :
            for entry in split :
                save_file(name, entry)
        return

    key, remaining_keys = split_keys[0], split_keys[1:]

    groups: dict[str, list[dict]] = dict()
    for entry in dataset :
        k = entry[key]
        if k not in groups :
            groups[k] = []
        groups[k].append(entry)
    
    for k, values in groups.items() :
        split_and_save_dataset(values, save_file, remaining_keys, train_dev_test, _seed)
    
# print_directory_tree("./data/datasets/TORGO")
# i = print_all_prompts("./data/datasets/TORGO")
# print(i)

