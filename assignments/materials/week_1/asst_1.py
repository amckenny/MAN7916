"""Python file to verify that the students' Python environment is working."""

import zipfile
from pathlib import Path

import nltk
import requests
import spacy


def download_models_corpora() -> None:
    """Download text models that will be used in the class"""
    print("Downloading text models...")
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("wordnet")
    if "en_core_web_sm" not in spacy.util.get_installed_models():
        spacy.cli.download("en_core_web_sm", False, False)

    print("Downloading Aussie Top 100 corpus")
    aussie_url = (
        "https://www.dropbox.com/scl/fi/f0u3n1gzz1ozd4xfww743/"
        "AussieTop100private.zip?rlkey=506ntkaetpute210sf79gwhz0&dl=1"
    )
    local_data_path = Path.cwd() / "local_data"
    assert local_data_path.exists()

    aussie_folder = local_data_path / "aussie"
    aussie_folder.mkdir(exist_ok=True)
    assert aussie_folder.exists()

    aussie_zip = aussie_folder / "AussieTop100private.zip"

    response = requests.get(aussie_url, stream=True, timeout=10000)
    if response.status_code == 200:
        with open(aussie_zip, "wb") as outfile:
            for chunk in response.iter_content(chunk_size=8192):
                outfile.write(chunk)
        print("Aussie corpus downloaded")
        with zipfile.ZipFile(aussie_zip, "r") as infile:
            infile.extractall(aussie_folder)
    else:
        print("Aussie corpus not downloaded - shoot me an email letting me know...")


def display_welcome_screen() -> None:
    """
    Displays a welcome screen when the program is executed.
    """
    print(
        "\n\n\n",
        "=" * 23,
        "\n = Welcome to MAN 7916 =\n",
        "=" * 23,
    )
    print("This program verifies that your Python environment is working.")


def validator(name_int: int) -> str:
    """
    Creates validation text for the students to verify their Python environment

    Args:
        name_int (int): an integer value derived from the user's name

    Returns:
        str: validation text
    """
    line = (name_int % 19) + 1
    position = name_int % 50
    path_to_file = Path.cwd()
    if "week_1" not in str(path_to_file):
        path_to_file = path_to_file / "assignments" / "materials" / "week_1"
    with open(path_to_file / "lipsum.txt", "r", encoding="utf8") as file:
        for _ in range(line):
            text = file.readline().split()[position]
    return text


def main() -> None:
    """
    Main function to verify that the students' Python environment is working.
    """
    display_welcome_screen()
    while True:
        name: str = input("Please type your name > ").strip()
        if name.isalpha() and len(name) > 1:
            break
        else:
            print("Please enter a valid name.")

    print(f"Hey {name}! Python seems to be working fine.")

    print(
        "Now we're going to download a few models and corpora"
        "that we will be working with this semester. This may take"
        "a few minues."
    )
    download_models_corpora()

    name_int = len(name) * 327
    validation_text = validator(name_int)
    print(
        "=" * 50,
        f"\nIn your assignment #1 submission, please include the following "
        f'text: "Python1-{name}-{validation_text}"\n',
        "=" * 50,
    )


if __name__ == "__main__":
    main()
