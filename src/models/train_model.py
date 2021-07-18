import pickle
import os

import pandas as pd
import click
from tqdm import tqdm

from src import config
from src.models.n_gram import CharacterNGram


@click.command()
@click.option("--context_sizes", "-cs", multiple=True,
              help="Context size used to train the model. More than one"
                   " context size can be used.")
def main(context_sizes):
    freq_array = pd.read_csv(config.RAW_NAMES_FILE_PATH)
    print("Training Models")
    for context_size in tqdm(context_sizes):
        context_size = int(context_size)
        character_n_gram = CharacterNGram(context_size=context_size)
        character_n_gram.fit(freq_array)
        file_name = f"n_gram_context_size_{context_size}.pkl"
        file_path = os.path.join(config.MODELS_DIR, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(character_n_gram, f)


if __name__ == "__main__":
    main()
