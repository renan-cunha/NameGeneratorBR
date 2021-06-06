from src import config
from typing import List

def char_to_idx(char: str) -> int:
    return config.ALPHABET.index(char)

def sentence_to_idx(string: str) -> List[int]:
    return [char_to_idx(x) for x in string]

def idx_to_char(idx: int) -> str:
    return config.ALPHABET[idx]