""" Script to set up NLTK environment by downloading necessary models """
# This script is intended to be run by man7916launcher, not directly.

import os
from pathlib import Path
import nltk


def main():
    """ Main function to set up NLTK environment """
    print("Getting NLTK Models", flush=True)
    nltk_dir = Path.cwd() / "models"
    assert nltk_dir.exists()
    os.environ["NLTK_DATA"] = str(nltk_dir)
    nltk.download("stopwords", quiet=True)
    nltk.download("vader_lexicon", quiet=True)


if __name__ == "__main__":
    main()
