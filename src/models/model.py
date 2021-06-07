import numpy as np
import pandas as pd

from src import config
from src import util
from typing import Set

class CharacterNGram:

    def __init__(self, num_context: int) -> None:
        self.num_context = num_context

    def fit(self, frequency_names: pd.DataFrame) -> None:
        self.add_word_borders(frequency_names)
        
        self.freq_array = self.create_freq_array()

        for name, count in frequency_names[['first_name', 'frequency_total']].itertuples(index=False):
            for index in range(self.num_context, len(name)):
                pair_idx = self.get_context_chair_idx(name, index)
                self.freq_array[pair_idx] += count

    def get_context_chair_idx(self, name, index):
        """Returns a tuple with the indices from the frequency array that
           corresponds to the context/name pair"""
        current_char = name[index]
        context = name[index-self.num_context:index]
        context_idx = util.sentence_to_idx(context)
        current_char_idx = util.char_to_idx(current_char)
        return tuple(context_idx + [current_char_idx])

    def create_freq_array(self) -> np.ndarray:
        size_alphabet = len(config.ALPHABET)
        char_freq_array_shape = tuple([size_alphabet for _ in range(self.num_context + 1)])
        return np.zeros(char_freq_array_shape)

    def add_word_borders(self, frequency_names: pd.DataFrame) -> None:
        prefix = config.START_CHAR * self.num_context
        first_name_col = frequency_names['first_name']
        frequency_names['first_name'] = prefix + first_name_col + config.END_CHAR

    def predict(self, context: str = "") -> str:
        result = config.START_CHAR * self.num_context + context.upper()
        while True:
            context_freq_array = self.get_context_freq_array(result)
            context_prob_array = context_freq_array / context_freq_array.sum()
            new_letter = np.random.choice(config.ALPHABET, 
                                          p=context_prob_array)             
            
            word_length = (len(result) - self.num_context)
            if new_letter != config.END_CHAR:
                result += new_letter
            elif word_length >= config.MINIMUM_LENGTH_NAME:
                break
        return result[self.num_context:]

    def get_context_freq_array(self, result):
        """Gets the frequency array for the most recent context"""
        if self.num_context > 0:
            context = result[-self.num_context:]
            context_idx = util.sentence_to_idx(context)
            context_freq_array = self.freq_array[tuple(context_idx)]
        else:
            context_freq_array = self.freq_array
        return context_freq_array

    def sample(self, number: int, max_attempts: int = None,
               context: str = "") -> Set[str]:
        
        result = set()
        attempts = 0
        while len(result) < number:
            new_name = self.predict(context=context)
            result.add(new_name)
            attempts += 1
            if max_attempts != None and attempts == max_attempts:
                break
        return result


if __name__ == "__main__":
    freq_array = pd.read_csv(config.RAW_NAMES_FILE_PATH)
    character_n_gram = CharacterNGram(num_context=3)
    character_n_gram.fit(freq_array)
    for i in character_n_gram.sample(5, context="phi"):
        print(i)
    