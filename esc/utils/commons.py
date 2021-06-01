import subprocess
from typing import List, Any, Iterable
from os import listdir
from os.path import isfile, isdir, join

import torch
from torch.nn.utils.rnn import pad_sequence


def flatten(lst: List[list]) -> list:
    return [_e for sub_l in lst for _e in sub_l]


def max_sublist_len(lst: List[list]) -> int:
    return max([len(x) for x in lst], default=0)


def nth_index_of(lst: Iterable[Any], item: Any, n: int) -> int:
    return [i for i, elem in enumerate(lst) if elem == item][n]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def list_elems_in_dir(dir_path: str, only_files: bool = False, only_dirs: bool = False) -> List[str]:

    elems_in_dir = [e for e in listdir(dir_path)]

    if only_files:
        return [e for e in elems_in_dir if isfile(join(dir_path, e))]

    if only_dirs:
        return [e for e in elems_in_dir if isdir(join(dir_path, e))]

    return elems_in_dir


def batch_data(sequences: List[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    return pad_sequence(sequences, batch_first=True, padding_value=pad_token_id)


def count_lines_in_file(path):
    return int(subprocess.check_output(f"wc -l \"{path}\"", shell=True).split()[0])
