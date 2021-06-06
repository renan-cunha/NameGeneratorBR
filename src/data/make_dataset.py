# -*- coding: utf-8 -*-
from pathlib import Path
import logging
import os

import click
import pandas as pd
from tqdm import tqdm

from dotenv import find_dotenv, load_dotenv
from src.data.transform_names_file import transform_names_file
from src.config import PROCESSED_FILE_PATH

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    csv_file_path = os.path.join(input_filepath, "nomes.csv")
    df = pd.read_csv(csv_file_path, dtype={"first_name": str, "frequency_total": int})
    df = df.sort_values("first_name")
    names_series = transform_names_file(df)

    with open(PROCESSED_FILE_PATH, "w", encoding="utf-8") as f:
        for name in tqdm(names_series):
            f.write(f"{name}\n")
            

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
