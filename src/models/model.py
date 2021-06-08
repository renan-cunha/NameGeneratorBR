import numpy as np
import pandas as pd

from src import config
from src import util
from typing import Set, Tuple

class CharacterNGram:

    def __init__(self, context_size: int, random_state: int = None) -> None:
        self.context_size = context_size
        self.freq_arrays = [self.create_freq_array(x) for x in range(self.context_size, -1, -1)]
        if random_state != None:
            np.random.seed(random_state)

    def fit(self, frequency_names: pd.DataFrame) -> None:
    
        for index, array in enumerate(self.freq_arrays):
            self.freq_arrays[index] = self.__populate_freq_array(array,
                                                                 frequency_names)

    def __populate_freq_array(self, array: np.ndarray,
                              frequency_names: pd.DataFrame) -> np.ndarray:
        
        context_size = len(array.shape) - 1
        frequency_names_copy = frequency_names.copy()
        frequency_names_copy = self.add_word_borders(frequency_names_copy,
                                                     context_size)
        
        for name, count in frequency_names_copy[['first_name', 'frequency_total']].itertuples(index=False):
            for index in range(context_size, len(name)):
                pair_idx = self.get_context_chair_idx(name, index, context_size)
                array[pair_idx] += count
        return array

    def get_context_chair_idx(self, name: str, index: int, 
                              context_size: int) -> Tuple[int]:
        """Returns a tuple with the indices from the frequency array that
           corresponds to the context/name pair"""
        current_char = name[index]
        context = name[index-context_size:index]
        context_idx = util.sentence_to_idx(context)
        current_char_idx = util.char_to_idx(current_char)
        return tuple(context_idx + [current_char_idx])

    def create_freq_array(self, context_size: int) -> np.ndarray:
        size_alphabet = len(config.ALPHABET)
        char_freq_array_shape = tuple([size_alphabet for _ in range(context_size + 1)])
        return np.zeros(char_freq_array_shape)

    def add_word_borders(self, frequency_names: pd.DataFrame,
                         context_size: int) -> pd.DataFrame:
        prefix = config.START_CHAR * context_size
        first_name_col = frequency_names['first_name']
        result_df = frequency_names.copy()
        result_df['first_name'] = prefix + first_name_col + config.END_CHAR
        return result_df

    def predict(self, context: str = "") -> str:
        
        if len(context) > 0 and not context.isalpha():
            raise ValueError("Context should contain only letters")         
        
        result = config.START_CHAR * self.context_size + context.upper()
        while True:
            context_freq_array = self.get_context_freq_array(result)
            context_prob_array = context_freq_array / context_freq_array.sum()
            new_letter = np.random.choice(config.ALPHABET, 
                                          p=context_prob_array)             
            
            word_length = (len(result) - self.context_size)
            if new_letter != config.END_CHAR:
                result += new_letter
            elif word_length >= config.MINIMUM_LENGTH_NAME:
                break
        return result[self.context_size:]

    def get_context_freq_array(self, result: str, 
                               back_off: int = 0) -> np.ndarray:
        """Gets the frequency array for the most recent context"""
        if self.context_size > 0:
            context = result[-self.context_size + back_off:]
            context_idx = util.sentence_to_idx(context)
            context_freq_array = self.freq_arrays[back_off][tuple(context_idx)]
            if np.any(context_freq_array) == False:
                print(f"{result}, {back_off}")
                return self.get_context_freq_array(result, back_off=back_off + 1)
        elif self.context_size == 0 or back_off == self.context_size:
            context_freq_array = self.freq_arrays[-1]
        return context_freq_array

    def sample(self, number: int, max_attempts: int = None,
               context: str = "") -> Set[str]:
        if max_attempts != None and max_attempts < number:
            raise ValueError(f"max_attemps shoud be greater than or equal to" 
                             f"number")        

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
    character_n_gram = CharacterNGram(context_size=4, random_state=42)
    character_n_gram.fit(freq_array)
    print(character_n_gram.sample(5, context="clo", max_attempts=10))
    