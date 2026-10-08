import warnings

import os
import sys
from datetime import datetime
from pathlib import Path

import nltk
import pyLDAvis
import spacy
import numpy as np
import pandas as pd
import scattertext as st
import torch
import tomotopy as tp

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.stats import pointbiserialr
from sklearn import linear_model, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, matthews_corrcoef
from torch.utils.data import DataLoader, Dataset, random_split

from scripts.dictionary.dict_analysis import read_dictionary, get_count

warnings.simplefilter("ignore", category=DeprecationWarning)

nltk_dir = Path.home() / "MAN7916" / "models"
assert nltk_dir.exists()
os.environ["NLTK_DATA"] = str(nltk_dir)
os.environ["OMP_NUM_THREADS"] = "16"

if not torch.cuda.is_available():
    print(
        "No GPU detected... You probably want to restart with a GPU-enabled runtime",
        flush=True,
    )
else:
    print(
        f"Found a GPU! Ready for the next step... ({torch.cuda.get_device_name(0)})",
        flush=True,
    )


corpora_path = Path.home() / "MAN7916" / "corpora"
dicts_path = Path.home() / "MAN7916" / "dictionaries"
dataset_dir = corpora_path / "aclImdb"
train_dir = dataset_dir / "train"
test_dir = dataset_dir / "test"
batch_size = 32
seed = 24601


class TextDataset(Dataset):
    def __init__(self, directory):
        self.data = []
        self.labels = []

        label_map = {"neg": 0, "pos": 1}
        self.class_names = list(label_map)

        for class_name, label in label_map.items():
            class_dir = Path(directory) / class_name

            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Required sentiment folder not found: {class_dir}"
                )

            for file_path in sorted(class_dir.glob("*.txt")):
                with open(file_path, "r", encoding="utf-8") as f:
                    self.data.append(f.read().strip())
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def dataset_to_dataframe(dataset):
    reviews, sentiments = zip(*dataset)
    return pd.DataFrame({"review": reviews, "sentiment": sentiments})


def coeff_of_imbalance(pos, neg, tot_words):
    if pos == neg or tot_words == 0:
        return 0.0

    denominator = (pos + neg) * tot_words

    if pos > neg:
        return (pos**2 - pos * neg) / denominator

    return (pos * neg - neg**2) / denominator


def parse_hyperparameter(name, text):
    text = text.strip()

    if name == "fit_prior":
        boolean_values = {
            "true": True,
            "yes": True,
            "1": True,
            "false": False,
            "no": False,
            "0": False,
        }
        if text.lower() not in boolean_values:
            raise ValueError("Enter true/false, yes/no, or 1/0.")
        return boolean_values[text.lower()]

    if name == "gamma" and text.lower() in {"auto", "scale"}:
        return text.lower()

    if name in {"C", "alpha", "gamma"}:
        value = float(text)
        if not np.isfinite(value) or value < 0 or (name == "C" and value == 0):
            requirement = "positive" if name == "C" else "non-negative"
            raise ValueError(f"{name} must be a finite {requirement} number.")
        return value

    if name in {"max_iter", "degree"}:
        value = int(text)
        minimum = 1 if name == "max_iter" else 0
        if value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
        return value

    choices = {
        "solver": {"lbfgs", "newton-cg", "liblinear", "sag", "saga"},
        "kernel": {"linear", "poly", "rbf", "sigmoid"},
    }
    if name in choices:
        value = text.lower()
        if value not in choices[name]:
            raise ValueError(f"Choose one of: {', '.join(sorted(choices[name]))}.")
        return value

    raise ValueError(f"Unsupported hyperparameter: {name}.")


print(f"\n====Loading Dataset==== - {datetime.now()}", flush=True)
print("""Loading the IMDB dataset: For simplicity, we're going to use a large
      publicly available dataset to demonstrate machine learning using multiple
      techniques. This dataset has already been loaded for you in the setup process.""")
full_train_dataset = TextDataset(train_dir)
test_dataset = TextDataset(test_dir)
class_names = full_train_dataset.class_names

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

full_train_data = dataset_to_dataframe(train_dataset)
full_validation_data = dataset_to_dataframe(val_dataset)
full_test_data = dataset_to_dataframe(test_dataset)

# Sample validation reviews for dictionary development and demonstrations.
development_data = full_validation_data.sample(
    n=3000,
    random_state=seed,
)
print(
    "Dictionary refinement uses development reviews. "
    "Scores on these reviews describe development performance."
)
development_data["txt_sent"] = development_data.apply(
    lambda x: "Positive" if x["sentiment"] == 1 else "Negative", axis=1
)
development_data["review"] = development_data["review"].apply(
    lambda x: x.replace("\\", "").replace("<br />", " ")
)

print(f"\n====Loading stopwords and dictionaries==== - {datetime.now()}", flush=True)
stops = nltk.corpus.stopwords.words("english") + ["'s", "&"]
dictionaries = {}
for file in dicts_path.glob("*.dict"):
    dictionary_data = read_dictionary(file)
    dictionaries[dictionary_data["var_name"]] = dictionary_data

print(
    f"\n====Preprocessing texts - this may take a while...==== - {datetime.now()}",
    flush=True,
)
nlp = spacy.load("en_core_web_sm")
docs = nlp.pipe(development_data["review"].tolist())
preprocessed = []
for doc in docs:
    preprocessed.append(
        [
            token.text.lower()
            for token in doc
            if token.pos_ not in ["PUNCT", "SYM", "NUM", "X"]
            and token.text.lower() not in stops
        ]
    )
development_data["review_tokens"] = preprocessed
_ = input("Preprocessing complete. Press Enter to continue...")
print("\n\n")
print("=" * 50)
print("Rules-based analyses as a point for comparison")
print("=" * 50)
print(
    "This data set links IMDB movie reviews to the overall 'sentiment'"
    " (whether it's a positive review or negative review) of the movie.\n\n"
    "Let's take a look at a few of the reviews.\n"
)
for _, row in development_data.head().iterrows():
    print(f"Review: {row['review'][:130]}...")
    print(f"Sentiment: {row['txt_sent']}\n")
print(
    "In dictionary-based coding, we look at the frequency with which a list"
    " of words thought to be associated with a construct (in this case, "
    "positive/negative sentiment) appear in a corpus of texts. The selection "
    "of words in the dictionaries is crucial for measure validity and "
    "reliability. In this case, we're looking at positive/negative words "
    "from Henry (2008).\nLet's take a look at these dictionaries:\n\n"
)
_ = input("Press Enter to continue...")
print(
    f"\n\n\n====Printing out all dictionaries' names, titles, and word lists.==== - {datetime.now()}",
    flush=True,
)
for name, dictionary in dictionaries.items():
    print(f"\nNAME: {name}")
    print(f"TITLE: {dictionary['title']}")
    print(f"WORDS: {', '.join(sorted(dictionary['words']))}\n{'-'*20}")
print(
    "\nThese words were selected for their relevance in business communications."
    " Yet our texts are movie reviews. We should probably see whether these "
    "dictionaries still make sense.\n\nLet's conduct a concordance analysis or "
    "a Key Word In Context (KWIC) analysis to see if these words have face "
    "validity: (Remember that stop words have been removed, so the english "
    "won't flow perfectly here...\n"
)
_ = input("Press Enter to continue...")
print("\n\n")
print(
    f"\n====Concordance Analysis==== - {datetime.now()}",
    flush=True,
)
look_for = "beat"
while True:
    print(f"Concordance for '{look_for}':\n")
    print(
        nltk.Text(
            [
                word
                for id, row in development_data.iterrows()
                for word in row["review_tokens"]
            ]
        ).concordance(look_for.lower())
    )
    if look_for == "beat":
        print(
            "\nChances are good that you found some instances that didn't seem quite "
            "right.\n\nWhy this might be bad: Language could differ in the current texts"
            " than in the texts from which these dictionaries were developed.\nWhy"
            " this might be OK: Finding some false-positives is inevitable even with "
            "the most appropriate dictionaries.\n\nWe try to minimize this, but it's a "
            "balancing act: the false negatives created by omitting a word that is "
            "usually used in-context creates measurement error variance as well. "
        )
    look_for = input(
        "\nEnter a word to look for in the concordance analysis (or press Enter to continue): "
    )
    if not look_for:
        break
print(
    "\n\n\nLet's hold our nose for a moment regarding some of these words and "
    "assume we're OK with the current dictionaries. Let's conduct the "
    "actual dictionary-based computer-aided text analysis:"
)
print(
    f"\n====Dictionary-based CATA for Henry (2008) positivity and negativity dictionaries==== - {datetime.now()}",
    flush=True,
)
development_data["positivity_henry_08"] = development_data["review_tokens"].apply(
    lambda x: get_count(x, dictionaries["Tone_Positivity_Henry08"]["words"])
)
development_data["negativity_henry_08"] = development_data["review_tokens"].apply(
    lambda x: get_count(x, dictionaries["Tone_Negativity_Henry08"]["words"])
)
print(development_data.head())
print(
    "\nNow that we have positivity and negativity scores, we can calculate a "
    "point-biserial correlation with the actual sentiment to see how accurate we are..."
)
print(
    f"\n====Calculating point-biserial correlations for positive and negative CATA with ground-truth sentiment==== - {datetime.now()}",
    flush=True,
)
pos_pbsr = pointbiserialr(
    x=development_data["sentiment"], y=development_data["positivity_henry_08"]
)
neg_pbsr = pointbiserialr(
    x=development_data["sentiment"], y=development_data["negativity_henry_08"]
)
print(
    f"The correlation between sentiment and the positivity dictionary is {pos_pbsr[0]:.02}; p = {pos_pbsr[1]:.03}"
)
print(
    f"The correlation between sentiment and the negativity dictionary is {neg_pbsr[0]:.02}; p = {neg_pbsr[1]:.03}"
)
print(
    "\nHowever, here we have two measures of sentiment rather than one. "
    "There are many ways the literature has combined such measures into a "
    "single sentiment score. A couple common approaches are:"
    "\n\n* Difference scores"
    "\n* Janis-Fadner coefficient of imbalance\n\n"
    "Let's look at how these overall sentiment scores correlate:\n"
)
development_data["sentiment_henry_08"] = (
    development_data["positivity_henry_08"] - development_data["negativity_henry_08"]
)
development_data["coeff_imb_henry_08"] = development_data.apply(
    lambda row: coeff_of_imbalance(
        row.positivity_henry_08, row.negativity_henry_08, len(row.review_tokens)
    ),
    axis=1,
)
sent_pbsr = pointbiserialr(
    x=development_data["sentiment"], y=development_data["sentiment_henry_08"]
)
coi_pbsr = pointbiserialr(
    x=development_data["sentiment"], y=development_data["coeff_imb_henry_08"]
)
print(
    f"The correlation between sentiment and the difference score is          {sent_pbsr[0]:.02}; p = {sent_pbsr[1]:.03}"
)
print(
    f"The correlation between sentiment and the coefficient of imbalance is  {coi_pbsr[0]:.02}; p = {coi_pbsr[1]:.03}"
)
_ = input("Press Enter to continue...")
print("\n\n")
print(
    "Chances are these word lists need refining... let's use the "
    "'scattertext' package to examine the distribution of words over "
    "positive/negative sentiment texts:"
)
print(f"\n====Creating a scattertext explorer plot==== - {datetime.now()}", flush=True)
st_corpus = (
    st.CorpusFromPandas(development_data, category_col="txt_sent", text_col="review")
    .build()
    .compact(st.AssociationCompactor(2000))
)
st_html = st.produce_scattertext_explorer(
    st_corpus,
    category="Positive",
    category_name="Positive",
    not_categories=["Negative"],
    sort_by_dist=False,
    term_scorer=st.CredTFIDF(st_corpus),
    background_color="#e5e5e3",
)
filepath = str(Path.home() / "MAN7916" / "output" / "scattertext.html")
with open(filepath, "w", encoding="utf-8") as outfile:
    outfile.write(st_html)
print(f"Scattertext plot saved to {filepath} - download it to your computer to view.")
print(
    "\nWe can use this chart to see words that are:\n\n"
    "* Used frequently in negative reviews and infrequently in positive "
    "reviews, but are not listed in our negative word list. (False Negatives)\n"
    "* Used frequently in positive reviews and infrequently in negative reviews,"
    " but are not listed in our positive word list. (False Negatives)\n"
    "* In our word lists, but do not discriminate well between positive and "
    "negative reviews. (False Positives)"
)
_ = input("Press Enter to continue...")
print("\n\nAs a reminder, here is what the Henry (2008) word lists contain:")
for name, dictionary in dictionaries.items():
    print(f"\nNAME: {name}")
    print(f"TITLE: {dictionary['title']}")
    print(f"WORDS: {', '.join(sorted(dictionary['words']))}\n{'-'*20}")
print(
    "\n\nACTIVITY\n\nUse the scattertext plot to add words to the "
    "positive/negative dictionaries based on:\n* How well they discriminate "
    "between positive and negative reviews\n* Whether you could theoretically "
    "justify their linkage to positive/negative sentiment (e.g., Just because "
    "'Seagal' tends to be in bad movies, doesn't mean we should use his name"
    " to influence negative sentiment scores.\n\nGo through the existing "
    "dictionaries and remove words that seem to cause problems in the "
    "dictionary-based analysis. \n"
)

add_to_positive = ["great", "brilliant", "perfect", "wonderful", "favorite", "loved"]
add_to_negative = ["worst", "terrible", "awful", "waste"]
remove_from_positive = [""]
remove_from_negative = ["declined"]
while True:
    print(f"\nCurrent words to add to positive: {', '.join(add_to_positive)}")
    new_positive = input(
        "Add a word to the positive dictionary (or press Enter to continue): "
    )
    if new_positive:
        add_to_positive.append(new_positive)
    else:
        break
print()
while True:
    print(f"\nCurrent words to add to negative: {', '.join(add_to_negative)}")
    new_negative = input(
        "Add a word to the negative dictionary (or press Enter to continue): "
    )
    if new_negative:
        add_to_negative.append(new_negative)
    else:
        break
print()
while True:
    print(f"\nCurrent words to remove from positive: {', '.join(remove_from_positive)}")
    remove_positive = input(
        "Remove a word from the positive dictionary (or press Enter to continue): "
    )
    if remove_positive:
        remove_from_positive.append(remove_positive)
    else:
        break
print()
while True:
    print(f"\nCurrent words to remove from negative: {', '.join(remove_from_negative)}")
    remove_negative = input(
        "Remove a word from the negative dictionary (or press Enter to continue): "
    )
    if remove_negative:
        remove_from_negative.append(remove_negative)
    else:
        break
print()
custom_pos = list(
    set(dictionaries["Tone_Positivity_Henry08"]["words"] + add_to_positive)
    - set(remove_from_positive)
)
custom_neg = list(
    set(dictionaries["Tone_Negativity_Henry08"]["words"] + add_to_negative)
    - set(remove_from_negative)
)
development_data["positivity_custom"] = development_data["review_tokens"].apply(
    lambda x: get_count(x, custom_pos)
)
development_data["negativity_custom"] = development_data["review_tokens"].apply(
    lambda x: get_count(x, custom_neg)
)
development_data["sentiment_custom"] = (
    development_data["positivity_custom"] - development_data["negativity_custom"]
)
development_data["coeff_imb_custom"] = development_data.apply(
    lambda row: coeff_of_imbalance(
        row.positivity_custom, row.negativity_custom, len(row.review_tokens)
    ),
    axis=1,
)
cus_pos_pbsr = pointbiserialr(
    x=development_data["sentiment"], y=development_data["positivity_custom"]
)
cus_neg_pbsr = pointbiserialr(
    x=development_data["sentiment"], y=development_data["negativity_custom"]
)
cus_sent_pbsr = pointbiserialr(
    x=development_data["sentiment"], y=development_data["sentiment_custom"]
)
cus_coi_pbsr = pointbiserialr(
    x=development_data["sentiment"], y=development_data["coeff_imb_custom"]
)

print(
    f"The correlation between sentiment and the positivity dictionary is......... ORIGINAL: {pos_pbsr[0]:.02}; p = {pos_pbsr[1]:.02} --- CUSTOM: {cus_pos_pbsr[0]:.02}; p = {cus_pos_pbsr[1]:.02}"
)
print(
    f"The correlation between sentiment and the negativity dictionary is......... ORIGINAL: {neg_pbsr[0]:.02}; p = {neg_pbsr[1]:.02} --- CUSTOM:{cus_neg_pbsr[0]:.02}; p = {cus_neg_pbsr[1]:.02}"
)
print(
    f"The correlation between sentiment and the difference score is.............. ORIGINAL: {sent_pbsr[0]:.02}; p = {sent_pbsr[1]:.02} --- CUSTOM: {cus_sent_pbsr[0]:.02}; p = {cus_sent_pbsr[1]:.02}"
)
print(
    f"The correlation between sentiment and the coefficient of imbalance is...... ORIGINAL: {coi_pbsr[0]:.02}; p = {coi_pbsr[1]:.02} --- CUSTOM: {cus_coi_pbsr[0]:.02}; p = {cus_coi_pbsr[1]:.02}"
)
_ = input("Press Enter to continue...")
print("\n\n")
print("=" * 50)
print("VADER (and other weighted rules-based sentiment analyses)")
print("=" * 50)
print(
    "VADER is an acronym for 'Valence Aware Dictionary and sEntiment Reasoner'"
    " and builds on basic dictionary-based computer-aided text anaylses by"
    " applying weights to sentiment words. For example, the word 'awesome' "
    "has a score of 3.1, whereas 'nice' has a score of 1.8, and 'horrible' "
    "has a score of -2.5. The VADER algorithm also explicitly addresses negation"
    " (e.g., 'isn't horrible') and boosting (e.g., 'very horrible') in a way "
    "that routine dictionary-based approaches do not.\n\nLet's take a look "
    "at some examples of what VADER produces for some sample phrases:\n"
)
vader_coder = SentimentIntensityAnalyzer()
good_phrase = "This is an awesome movie"
bad_phrase = "This is a horrible movie"
negated_bad = "This isn't a horrible movie"
boosted_bad = "This is a very horrible movie"

print(f"'{good_phrase}' is coded as {vader_coder.polarity_scores(good_phrase)}")
print(f"'{bad_phrase}' is coded as {vader_coder.polarity_scores(bad_phrase)}")
print(f"'{negated_bad}' is coded as {vader_coder.polarity_scores(negated_bad)}")
print(f"'{boosted_bad}' is coded as {vader_coder.polarity_scores(boosted_bad)}")
print(
    "\nOn face, this seems a significant improvement over what we would see"
    " with a generic dictionary-based computer-aided text analysis.\nLet's see "
    "how well it compares quantitatively on our corpus of IMDB movie reviews:\n"
)
development_data["vader_comp"] = development_data["review"].apply(
    lambda x: vader_coder.polarity_scores(x)["compound"]
)
vader_pbsr = pointbiserialr(
    x=development_data["sentiment"], y=development_data["vader_comp"]
)
print(
    f"The correlation between sentiment and the VADER score is................... ORIGINAL: {vader_pbsr[0]:.02}; p = {vader_pbsr[1]:.02}"
)
print(f"{'-'*120}")
print(
    f"The correlation between sentiment and the difference score is.............. ORIGINAL: {sent_pbsr[0]:.02}; p = {sent_pbsr[1]:.02} --- CUSTOM: {cus_sent_pbsr[0]:.02}; p = {cus_sent_pbsr[1]:.02}"
)
print(
    f"The correlation between sentiment and the coefficient of imbalance is...... ORIGINAL: {coi_pbsr[0]:.02}; p = {coi_pbsr[1]:.02} --- CUSTOM: {cus_coi_pbsr[0]:.02}; p = {cus_coi_pbsr[1]:.02}"
)

print(
    "\n\nClearly the original Henry (2008) dictionaries didn't fare well against "
    "VADER - but that wasn't a fair comparison.\n\nIn contrast, if you were "
    "thorough in your refinement of the positive/negative dictionaries, you "
    "may well have matched or even surpassed the correlation from VADER.\nVADER "
    "was developed using social media texts. So while perhaps closer to movie "
    "reviews than the business texts used by Henry, even the more nuanced "
    "approach to coding used by VADER may struggle to outperform a dictionary "
    "custom-developed to your context.\n"
)
_ = input("Press Enter to continue...")
print("\n\n")
print("=" * 50)
print("Supervised Machine Learning")
print("=" * 50)
print(
    "Dictionary-/rules-based approaches are particularly valuable when you "
    "do not have a large number of already classified texts. This is often "
    "the case in our research, where the reason many business scholars are "
    "interested in CATA is because we do not have access to the 'ground "
    "truth' for a large number of observations. However, as ground-truth "
    "observations become available, we can use sequence classification "
    "algorithms in machine learning to have the computer learn to distinguish "
    "between positive and negative sentiment in text. However, before we can "
    "do that, we must first figuratively teach the computer to read."
)
vectorizer = TfidfVectorizer(max_features=10000)
x_train_tfidf = vectorizer.fit_transform(full_train_data["review"])
x_validation_tfidf = vectorizer.transform(full_validation_data["review"])

print(
    "\n\nLogistic Regression: This technique should sound familiar. This "
    "workhorse of the statistical world reappears in the machine learning "
    "world as a basic model for classification. Here we're regressing the "
    "'ground truth' sentiment on the words used in the text (expressed as "
    "a tf-idf vector)."
)
hyperparameters = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 100,
}
while True:
    print(f"\n====~~~~~HYPERPARAMETERS~~~~~==== - {datetime.now()}", flush=True)
    print(hyperparameters)

    print(
        f"\n====Train the logistic regression classifier==== - {datetime.now()}",
        flush=True,
    )
    lr_classifier = linear_model.LogisticRegression(
        C=hyperparameters["C"],
        solver=hyperparameters["solver"],
        max_iter=hyperparameters["max_iter"],
    )
    lr_classifier.fit(x_train_tfidf, full_train_data["sentiment"])
    lr_predictions = lr_classifier.predict(x_validation_tfidf)

    print(
        f"\n====Estimate and print the accuracy and phi coefficient for the logistic regression classifier==== - {datetime.now()}",
        flush=True,
    )
    lr_accuracy = accuracy_score(lr_predictions, full_validation_data["sentiment"])
    lr_phi = matthews_corrcoef(lr_predictions, full_validation_data["sentiment"])

    print(f"Logistic Regression accuracy: {lr_accuracy:.2%}")
    print(f"Logistic Regression phi coefficient (correlation): {lr_phi:.02}")

    print("\n\nWould you like to try different hyperparameters?")
    if input("Enter 'y' to try different hyperparameters: ").lower() != "y":
        break
    while True:
        print("Which hyperparameter would you like to change?")
        print("\n".join(hyperparameters.keys()))
        hyperparameter = input("Enter the hyperparameter you would like to change: ")
        if hyperparameter not in hyperparameters:
            print("Invalid hyperparameter. Please try again.")
            continue
        new_value = input(f"Enter the new value for {hyperparameter}: ")
        try:
            hyperparameters[hyperparameter] = parse_hyperparameter(
                hyperparameter, new_value
            )
        except ValueError as error:
            print(f"Invalid value: {error} Please try again.")
            continue
        print("Change another hyperparameter?")
        if input("Enter 'y' to change another hyperparameter: ").lower() != "y":
            break


print(
    "\n\nNaïve Bayes: Like with logistic regression, you have likely worked with "
    "a foundational component of the naïve Bayes classifier in statistics."
    " This classifier uses a key insight from Bayesian statistics to use words"
    " to predict a classification. Specifically, we're going to use Bayes' "
    "rule to find the P(classification|words) given the "
    "P(words|classification), P(classification), and P(words)."
)
hyperparameters = {"alpha": 1.00, "fit_prior": True}
while True:
    print(f"\n====~~~~~HYPERPARAMETERS~~~~~==== - {datetime.now()}", flush=True)
    print(hyperparameters)

    print(f"\n====Train the naive bayes classifier==== - {datetime.now()}", flush=True)
    nb_classifier = naive_bayes.MultinomialNB(
        alpha=hyperparameters["alpha"], fit_prior=hyperparameters["fit_prior"]
    )
    nb_classifier.fit(x_train_tfidf, full_train_data["sentiment"])
    nb_predictions = nb_classifier.predict(x_validation_tfidf)

    print(
        f"\n====Estimate and print the accuracy and phi coefficient for the naive bayes classifier==== - {datetime.now()}",
        flush=True,
    )
    nb_accuracy = accuracy_score(nb_predictions, full_validation_data["sentiment"])
    nb_phi = matthews_corrcoef(nb_predictions, full_validation_data["sentiment"])

    print(f"Naive Bayes' accuracy: {nb_accuracy:.2%}")
    print(f"Naive Bayes' phi coefficient (correlation): {nb_phi:.02}")

    print("\n\nWould you like to try different hyperparameters?")
    if input("Enter 'y' to try different hyperparameters: ").lower() != "y":
        break
    while True:
        print("Which hyperparameter would you like to change?")
        print("\n".join(hyperparameters.keys()))
        hyperparameter = input("Enter the hyperparameter you would like to change: ")
        if hyperparameter not in hyperparameters:
            print("Invalid hyperparameter. Please try again.")
            continue
        new_value = input(f"Enter the new value for {hyperparameter}: ")
        try:
            hyperparameters[hyperparameter] = parse_hyperparameter(
                hyperparameter, new_value
            )
        except ValueError as error:
            print(f"Invalid value: {error} Please try again.")
            continue
        print("Change another hyperparameter?")
        if input("Enter 'y' to change another hyperparameter: ").lower() != "y":
            break

print(
    "\n\nRandom Forest: The random forest classifier is what is called an "
    "'ensemble' classifier. That is, the random forest classifier actually "
    "solicits the classifications from a number of other classifiers and "
    "treats them as 'votes', tallies the votes and produces a classification "
    "on that basis. It's a little more complicated than that, but this is the "
    "basic idea. The reason it is called a 'random forest'? Well, the "
    "classifiers is solicits votes from are 'decision trees.'"
)
hyperparameters = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 500,
    "min_samples_split": 2,  # Minimum number of texts in the branch when a split is made - integer (technically you can have a float, but stick with integer)
    "min_samples_leaf": 1,  # Minimum number of texts in a leaf on the decision tree - integer (technically you can have a float, but stick with integer)
}
while True:
    print(f"\n====~~~~~HYPERPARAMETERS~~~~~==== - {datetime.now()}", flush=True)
    print(hyperparameters)

    print(
        f"\n====Train the random forest classifier==== - {datetime.now()}", flush=True
    )
    rf_classifier = RandomForestClassifier(
        n_estimators=hyperparameters["n_estimators"],
        criterion=hyperparameters["criterion"],
        max_depth=hyperparameters["max_depth"],
        min_samples_split=hyperparameters["min_samples_split"],
        min_samples_leaf=hyperparameters["min_samples_leaf"],
        n_jobs=-1,
    )
    rf_classifier.fit(x_train_tfidf, full_train_data["sentiment"])
    rf_predictions = rf_classifier.predict(x_validation_tfidf)

    print(
        f"\n====Estimate and print the accuracy and phi coefficient for the random forest classifier==== - {datetime.now()}",
        flush=True,
    )
    rf_accuracy = accuracy_score(rf_predictions, full_validation_data["sentiment"])
    rf_phi = matthews_corrcoef(rf_predictions, full_validation_data["sentiment"])

    print(f"Random Forest accuracy: {rf_accuracy:.2%}")
    print(f"Random Forest phi coefficient (correlation): {rf_phi:.02}")

    print("\n\nWould you like to try different hyperparameters?")
    if input("Enter 'y' to try different hyperparameters: ").lower() != "y":
        break
    while True:
        print("Which hyperparameter would you like to change?")
        print("\n".join(hyperparameters.keys()))
        hyperparameter = input("Enter the hyperparameter you would like to change: ")
        if hyperparameter not in hyperparameters:
            print("Invalid hyperparameter. Please try again.")
            continue
        new_value = input(f"Enter the new value for {hyperparameter}: ")
        try:
            if hyperparameter == "criterion" and new_value not in ["gini", "entropy"]:
                print("Invalid value. Please try again.")
                continue
            if (
                hyperparameter == "n_estimators"
                or hyperparameter == "max_depth"
                or hyperparameter == "min_samples_split"
                or hyperparameter == "min_samples_leaf"
            ) and (not new_value.isdigit() or int(new_value) < 1):
                print("Invalid value. Please try again.")
                continue
            hyperparameters[hyperparameter] = type(hyperparameters[hyperparameter])(
                new_value
            )
        except ValueError:
            print("Invalid value. Please try again.")
            continue
        print("Change another hyperparameter?")
        if input("Enter 'y' to change another hyperparameter: ").lower() != "y":
            break


print(
    "\n\nSupport Vector Machine: The support vector machine has become a very"
    " popular classifier for its flexibility. It handles high feature/sample"
    " size ratios well, supports several kernel functions for when the "
    "decision boundary between two sets are not linearly separable, and is"
    " pretty memory efficient for what it does. It can, however, be quite "
    "slow... be prepared to wait on this one for a while..."
)
hyperparameters = {
    "C": 1.0,
    "kernel": "linear",
    "degree": 3,
    "gamma": "auto",
}
while True:
    print(f"\n====~~~~~HYPERPARAMETERS~~~~~==== - {datetime.now()}", flush=True)
    print(hyperparameters)

    print(
        f"\n====Train the support vector machine classifier==== - {datetime.now()}",
        flush=True,
    )
    svm_classifier = svm.SVC(
        C=hyperparameters["C"],
        kernel=hyperparameters["kernel"],
        degree=hyperparameters["degree"],
        gamma=hyperparameters["gamma"],
    )
    svm_classifier.fit(x_train_tfidf, full_train_data["sentiment"])
    svm_predictions = svm_classifier.predict(x_validation_tfidf)

    print(
        f"\n====Estimate and print the accuracy and phi coefficient for the random forest classifier==== - {datetime.now()}",
        flush=True,
    )
    svm_accuracy = accuracy_score(svm_predictions, full_validation_data["sentiment"])
    svm_phi = matthews_corrcoef(svm_predictions, full_validation_data["sentiment"])

    print(f"Support Vector Machine accuracy: {svm_accuracy:.2%}")
    print(f"Support Vector Machine phi coefficient (correlation): {svm_phi:.02}")

    print("\n\nWould you like to try different hyperparameters?")
    if input("Enter 'y' to try different hyperparameters: ").lower() != "y":
        break
    while True:
        print("Which hyperparameter would you like to change?")
        print("\n".join(hyperparameters.keys()))
        hyperparameter = input("Enter the hyperparameter you would like to change: ")
        if hyperparameter not in hyperparameters:
            print("Invalid hyperparameter. Please try again.")
            continue
        new_value = input(f"Enter the new value for {hyperparameter}: ")
        try:
            hyperparameters[hyperparameter] = parse_hyperparameter(
                hyperparameter, new_value
            )
        except ValueError as error:
            print(f"Invalid value: {error} Please try again.")
            continue
        print("Change another hyperparameter?")
        if input("Enter 'y' to change another hyperparameter: ").lower() != "y":
            break

print("\nDICTIONARY DEVELOPMENT PERFORMANCE")
print("These correlations use reviews inspected during dictionary refinement.")

for name, original, refined in [
    ("Positivity", pos_pbsr, cus_pos_pbsr),
    ("Negativity", neg_pbsr, cus_neg_pbsr),
    ("Difference score", sent_pbsr, cus_sent_pbsr),
    ("Imbalance coefficient", coi_pbsr, cus_coi_pbsr),
]:
    print(f"{name}: original r={original[0]:.3f}; " f"refined r={refined[0]:.3f}")

print(f"VADER development correlation: {vader_pbsr[0]:.3f}")

print("\nFINAL HELD-OUT TEST PERFORMANCE")
x_test_tfidf = vectorizer.transform(full_test_data["review"])
test_labels = full_test_data["sentiment"]

for name, classifier in [
    ("Logistic Regression", lr_classifier),
    ("Naive Bayes", nb_classifier),
    ("Random Forest", rf_classifier),
    ("Support Vector Machine", svm_classifier),
]:
    test_predictions = classifier.predict(x_test_tfidf)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_mcc = matthews_corrcoef(test_labels, test_predictions)
    print(f"{name}: accuracy={test_accuracy:.2%}; " f"MCC={test_mcc:.3f}")
_ = input("Press Enter to continue...")
print("\n\n")
print("=" * 50)
print("Unsupervised Machine Learning")
print("=" * 50)
print(
    "With supervised machine learning we provided the algorithm with both the "
    "inputs and desired output. Specifically, we used a 'classification' "
    "algorithm to use text inputs to predict a categorical output (i.e., "
    "sentiment). In unsupervised machine learning algorithms, we provide "
    "the algorithm with the inputs, but we do not provide it with a desired "
    "output.\nTo provide an analogy to statistical techniques we use in "
    "academia:\n* Supervised machine learning is like OLS, logistic regression, "
    "and SEM - We provide both X and Y variables and the algorithm figures out "
    "the best way to link them based on predetermined rules. "
    "\n* Unsupervised machine learning is like exploratory factor analysis, "
    "principal components analysis, and cluster analysis - We provide only X "
    "variables and the algorithm determines how to combine those X variables "
    "in ways that help us understand more about those variables.\n\n"
)
print("=" * 50)
print("Topic Modeling")
print("=" * 50)
print(
    "There are many unsupervised machine learning algorithms, but a popular "
    "one in management research right now is topic modeling.\nTopic models are "
    "designed to discover the 'topics' that occur in a corpus of documents. "
    "These models assume that documents are mixtures of topics, which "
    "themselves are characterized as a distribution over words. To provide "
    "a bit of an oversimplification, but one which academics might find "
    "accessible, topic models are a bit like a factor analysis of words. "
    "Words that 'hang together' frequently become associated with a latent "
    "'topic'.\n\n"
)
print(
    "Latent Dirichlet Allocation: Latent Dirichlet Allocation (LDA) is a "
    "foundational topic modeling technique used in Natural Language Processing. "
    "When applying LDA, a researcher specifies the number of topics to be "
    "extracted from the corpus a priori (k), and the algorithm then iteratively "
    "learns the word distributions for each topic and the topic distributions "
    "for each document. The below code uses LDA to uncover topics associated "
    "with the IMDB dataset we have been working with.\n"
)
hyperparameters = {
    "k": 10,
    "term_weight": tp.TermWeight.ONE,
    "min_cf": 3,
    "min_df": 1,
    "rm_top": 5,
    "alpha": 0.1,
    "eta": 0.01,
}
while True:
    print(f"\n====~~~~~HYPERPARAMETERS~~~~~==== - {datetime.now()}", flush=True)
    print(hyperparameters)

    print(f"\n====Train Latent Dirichlet Allocation==== - {datetime.now()}", flush=True)
    model = tp.LDAModel(
        tw=hyperparameters["term_weight"],
        min_cf=hyperparameters["min_cf"],
        min_df=hyperparameters["min_df"],
        rm_top=hyperparameters["rm_top"],
        alpha=hyperparameters["alpha"],
        eta=hyperparameters["eta"],
        k=hyperparameters["k"],
    )
    for review_text in development_data["review_tokens"]:
        model.add_doc(review_text)
    model.burn_in = 100
    model.train(0)
    print(
        "Number of documents:",
        len(model.docs),
        ", Vocab size:",
        len(model.used_vocabs),
        ", Number of words:",
        model.num_words,
    )
    print("Removed top words:", model.removed_top_words)
    print("Training...", file=sys.stderr, flush=True)
    model.train(1000)

    topic_term_dists = np.stack([model.get_topic_word_dist(k) for k in range(model.k)])
    doc_topic_dists = np.stack([doc.get_topic_dist() for doc in model.docs])
    doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
    doc_lengths = np.array([len(doc.words) for doc in model.docs])
    vocab = list(model.used_vocabs)
    term_frequency = model.used_vocab_freq

    for k in range(model.k):
        print(f"Topic #{k}")
        for word, prob in model.get_topic_words(k):
            print("\t", word, prob, sep="\t")

    print("\n\nWould you like to try different hyperparameters?")
    if input("Enter 'y' to try different hyperparameters: ").lower() != "y":
        break
    while True:
        print("Which hyperparameter would you like to change?")
        print("\n".join(hyperparameters.keys()))
        hyperparameter = input("Enter the hyperparameter you would like to change: ")
        if hyperparameter not in hyperparameters:
            print("Invalid hyperparameter. Please try again.")
            continue
        new_value = input(f"Enter the new value for {hyperparameter}: ").strip()

        try:
            if hyperparameter == "term_weight":
                term_weights = {
                    "one": tp.TermWeight.ONE,
                    "pmi": tp.TermWeight.PMI,
                    "idf": tp.TermWeight.IDF,
                }
                weight_name = new_value.lower()
                if weight_name not in term_weights:
                    raise ValueError("Choose one, pmi, or idf.")
                parsed_value = term_weights[weight_name]

            elif hyperparameter in {"k", "min_cf", "min_df", "rm_top"}:
                parsed_value = int(new_value)
                minimum = 1 if hyperparameter == "k" else 0
                if parsed_value < minimum:
                    raise ValueError(
                        f"{hyperparameter} must be an integer >= {minimum}."
                    )

            elif hyperparameter in {"alpha", "eta"}:
                parsed_value = float(new_value)
                if not np.isfinite(parsed_value) or parsed_value <= 0:
                    raise ValueError(
                        f"{hyperparameter} must be a finite positive number."
                    )

            else:
                raise ValueError("Unsupported hyperparameter.")

            hyperparameters[hyperparameter] = parsed_value

        except ValueError as error:
            print(f"Invalid value: {error} Please try again.")
            continue
        print("Change another hyperparameter?")
        if input("Enter 'y' to change another hyperparameter: ").lower() != "y":
            break

print(f"\n====Generate pyLDAvis plot==== - {datetime.now()}", flush=True)
prepared_data = pyLDAvis.prepare(
    topic_term_dists,
    doc_topic_dists,
    doc_lengths,
    vocab,
    term_frequency,
    start_index=1,
    sort_topics=False,
)
filename = str(Path.home() / "MAN7916" / "output" / "pyLDAvis_LDA.html")
pyLDAvis.save_html(prepared_data, filename)
print(f"pyLDAvis plot saved to {filename} - download it to your computer to view.")

print("Done!")
