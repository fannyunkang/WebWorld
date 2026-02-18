import random
import numpy as np
import torch
import hashlib


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_unique_id(task_path: str, sample_index: int) -> str:
    unique_id = f"{task_path}_{sample_index}"
    return unique_id


def generate_seed_from_id(unique_id: str) -> int:
    hash_object = hashlib.sha256(unique_id.encode())
    return int(hash_object.hexdigest(), 16) % (2**32)
