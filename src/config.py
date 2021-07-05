import os

START_CHAR = "_"
END_CHAR = "."
ALPHABET = [START_CHAR] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [END_CHAR]
UNIGRAM_FILE_PATH = os.path.join("data", "processed", "unigrams.txt")
BIGRAM_FILE_PATH = os.path.join("data", "processed", "bigrams.npy")
RAW_NAMES_FILE_PATH = os.path.join("data", "raw", "nomes.csv")
MODELS_DIR = "models"
MINIMUM_LENGTH_NAME = 3