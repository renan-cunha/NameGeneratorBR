from typing import Set

import numpy as np
import pandas as pd
import pickle

from src import config
from src import util


class CharacterNGram:

    def __init__(self, context_size: int) -> None:
        self.context_size = context_size
        context_sizes = range(self.context_size, -1, -1)
        self.freq_arrays = [self.create_freq_array(x) for x in context_sizes]

    def fit(self, freq_names: pd.DataFrame) -> None:

        for index, array in enumerate(self.freq_arrays):
            self.freq_arrays[index] = self.__populate_freq_array(array,
                                                                 freq_names)

    def __populate_freq_array(self, array: np.ndarray,
                              frequency_names: pd.DataFrame) -> np.ndarray:

        context_size = len(array.shape) - 1
        frequency_names_copy = frequency_names[['first_name',
                                                'frequency_total']].copy()
        frequency_names_copy = self.add_word_borders(frequency_names_copy,
                                                     context_size)

        for name, count in frequency_names_copy.itertuples(index=False):
            for index in range(context_size, len(name)):
                pair_idx = self.get_context_chair_idx(name, index,
                                                      context_size)
                array[pair_idx] += count
        return array

    def get_context_chair_idx(self, name: str, index: int,
                              context_size: int) -> tuple:
        """Returns a tuple with the indices from the frequency array that
           corresponds to the context/name pair"""
        current_char = name[index]
        context = name[index-context_size:index]
        context_idx = util.sentence_to_idx(context)
        current_char_idx = util.char_to_idx(current_char)
        return tuple(context_idx + [current_char_idx])

    def create_freq_array(self, context_size: int) -> np.ndarray:
        size_alphabet = len(config.ALPHABET)
        char_freq_array_shape = tuple(
            [size_alphabet for _ in range(context_size + 1)]
            )
        return np.zeros(char_freq_array_shape, dtype="uintc")

    def add_word_borders(self, frequency_names: pd.DataFrame,
                         context_size: int) -> pd.DataFrame:
        prefix = config.START_CHAR * context_size
        first_name_col = frequency_names['first_name']
        result_df = frequency_names.copy()
        result_df['first_name'] = prefix + first_name_col + config.END_CHAR
        return result_df

    def get_word_probability(self, word: str) -> float:
        """Gets the probability that the N-Gram model assigns to the word"""
        word = self.get_start_char() + word + config.END_CHAR
        word = word.upper()
        probability = 0
        for index, letter in enumerate(word[self.context_size:]):
            context = word[index: index + self.context_size]
            back_off = 0
            letter_idx = util.char_to_idx(letter)
            letter_context_prob = self.get_letter_context_prob(context,
                                                               back_off,
                                                               letter_idx)
            probability += np.log(letter_context_prob)
        return np.exp(probability)

    def get_letter_context_prob(self, context, back_off, letter_idx):
        while True:
            context_freq_array = self.get_context_freq_array(context,
                                                             back_off=back_off)
            context_prob_array = context_freq_array / context_freq_array.sum()
            letter_context_probability = context_prob_array[letter_idx]
            if letter_context_probability != 0:
                break
            back_off += 1
        if back_off > 0:
            letter_context_probability * 0.4 * back_off
        return letter_context_probability

    def predict(self, prefix: str = "", random_state: int = None) -> str:
        if random_state is not None:
            np.random.seed(random_state)

        if len(prefix) > 0 and not prefix.isalpha():
            raise ValueError("Context should contain only letters")

        result = self.get_start_char() + prefix.upper()
        while True:
            context_freq_array = self.get_context_freq_array(result)
            context_prob_array = context_freq_array / context_freq_array.sum()
            new_letter = np.random.choice(config.ALPHABET,
                                          p=context_prob_array)

            word_length = (len(result) - self.context_size)
            if new_letter != config.END_CHAR:
                result += new_letter
                if word_length > config.MAXIMUM_LENGTH_NAME:
                    break
            elif word_length >= config.MINIMUM_LENGTH_NAME:
                break

        return result[self.context_size:]

    def get_start_char(self) -> str:
        """Returns the blanking start char to add in the beginning of the
        word"""
        return config.START_CHAR * self.context_size

    def get_context_freq_array(self, result: str,
                               back_off: int = 0) -> np.ndarray:
        """Gets the frequency array for the most recent context"""
        if self.context_size == 0 or back_off == self.context_size:
            context_freq_array = self.freq_arrays[-1]
        else:
            context = result[-self.context_size + back_off:]
            context_idx = util.sentence_to_idx(context)
            context_freq_array = self.freq_arrays[back_off][tuple(context_idx)]
            if np.any(context_freq_array) is False:
                print(f"{result}, {back_off}")
                return self.get_context_freq_array(result,
                                                   back_off=back_off + 1)
        return context_freq_array

    def sample(self, number: int, max_attempts: int = None,
               context: str = "") -> Set[str]:
        if max_attempts is not None and max_attempts < number:
            raise ValueError("max_attemps shoud be greater than or equal to"
                             "number")

        result: Set[str] = set()
        attempts = 0
        while len(result) < number:
            new_name = self.predict(prefix=context)
            result.add(new_name)
            attempts += 1
            if max_attempts is not None and attempts == max_attempts:
                break
        return result


if __name__ == "__main__":
    with open("models/n_gram_context_size_1.pkl", 'rb') as f:
        model = pickle.load(f)
    while True:
        print(model.predict(random_state=42))
