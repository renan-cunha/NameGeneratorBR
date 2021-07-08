import click
import pickle
import os

from src import config
from src.models.train_model import CharacterNGram

@click.command()
@click.option("--context_size", "-cs", type=int, help=f"How much context to use for the" 
              " language model, The pre-trained models go from 0 to 4")
@click.option("--prefix", "-p", default="", type=str,
              help="The beginning of the name to be predicted (OPTIONAL)")
@click.option("--seed", '-s', default=None, type=int,
              help="Seed to reproduce experiments (OPTIONAL)")
def main(context_size, prefix, seed):
    file_path = os.path.join(config.MODELS_DIR, 
                             f"n_gram_context_size_{context_size}.pkl")
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
    except:
        raise FileNotFoundError(f"No model trained with context size equal to {context_size}")
    name = model.predict(prefix=prefix, random_state=seed)
    print(f"Predicted the name: {name}\n"
          f"Prefix: {prefix.upper() if prefix else 'None'}\n"
          f"Context Size: {int(context_size)}\n"
          f"Seed: {seed if seed is not None else 'RANDOM'}")
    
    
if __name__ == "__main__":
    main()