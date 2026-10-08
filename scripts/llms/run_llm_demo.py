import evaluate
import numpy as np
from pprint import pprint
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    pipeline,
    TrainingArguments,
    Trainer,
)

TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")
ACCURACY = evaluate.load("accuracy")


def prompt_user_to_continue():
    print("Press Enter to continue...")
    input()
    print("-" * 25, "\n")


def preprocess_function(examples):
    return TOKENIZER(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return ACCURACY.compute(predictions=predictions, references=labels)


def main():
    print("Step 1: Loading the IMDB dataset.")
    print(
        "This dataset contains movie reviews along with sentiment labels (0 for negative, 1 for positive)."
    )
    print("Loading IMDB dataset... This may take a minute.")
    imdb = load_dataset("imdb")

    training_split = imdb["train"].train_test_split(
        test_size=0.2,
        seed=24601,
        stratify_by_column="label",
    )

    imdb = DatasetDict({
        "train": training_split["train"],
        "validation": training_split["test"],
        "test": imdb["test"],
    })
    print("-" * 25, "Raw Sample Data", "-" * 25)
    pprint(imdb["test"][0], width=100, indent=5, compact=True)
    prompt_user_to_continue()

    print("Step 2: Tokenizing the dataset using the DistilBERT tokenizer.")
    print(
        "Tokenization converts raw text into tokens (numerical values) that the model can understand."
    )
    tokenized_imdb = imdb.map(
        preprocess_function, batched=True, remove_columns=["text"]
    )
    print("-" * 25, "Tokenized Sample", "-" * 25)
    pprint(tokenized_imdb["test"][0], width=100, indent=5, compact=True)
    prompt_user_to_continue()

    print("Step 3: Setting up the data collator for dynamic padding.")
    print(
        "The data collator pads sequences in a batch to the length of the longest sequence, which saves memory."
    )
    data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)
    sample_batch = data_collator([tokenized_imdb["test"][0], tokenized_imdb["test"][1]])
    print("Data collator output for a sample batch:")
    pprint(sample_batch, width=100, indent=5, compact=True)
    prompt_user_to_continue()

    print("Step 4: Initializing the DistilBERT model for sequence classification.")
    print(
        "DistilBERT is a smaller, faster version of BERT. "
        "We start with the pretrained distilbert-base-uncased checkpoint "
        "and add a newly initialized classification head for positive "
        "and negative sentiment. This exercise fine-tunes both the "
        "pretrained model and the classification head on labeled "
        "IMDB reviews."
    )
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
    )
    print("Model architecture summary:")
    print(model)
    prompt_user_to_continue()

    print("Step 5: Preparing training arguments and Trainer.")
    print(
        "For a quick demonstration, enter 50 to use 50 training reviews "
        "and 50 validation reviews. Validation runs after each epoch "
        "and determines which checkpoint is selected. "
        "After training, the selected checkpoint is evaluated once on "
        "all 25,000 official test reviews. That final evaluation may "
        "take substantial time even when the training sample is small."
    )
    training_args = TrainingArguments(
        output_dir="imdb_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
    )

    while True:
        print("\nHow many training reviews would you like to use?")
        print(
            "Validation will use the same number of reviews, "
            "up to the available validation-set size."
        )
        print(
            "Enter a positive whole number, "
            "or -1 to use all training and validation reviews."
        )
        user_input = input("Training review count: ").strip()
        if user_input.strip() == "-1":
            train_subset = tokenized_imdb["train"].shuffle(seed=24601)
            break
        elif user_input.isdigit() and int(user_input) > 0 and int(user_input) <= len(tokenized_imdb["train"]):
            num_samples = int(user_input)
            train_subset = tokenized_imdb["train"].shuffle(seed=24601).select(range(num_samples))
            break
        elif user_input.isdigit() and int(user_input) > 0 and int(user_input) > len(tokenized_imdb["train"]):
            print("Number of samples exceeds the total number of training samples.")
            print("Defaulting to all training samples.")
            train_subset = tokenized_imdb["train"].shuffle(seed=24601)
            break
        else:
            print("Invalid input. Please enter a valid number")

    validation_pool = tokenized_imdb["validation"].shuffle(seed=24601)
    eval_subset = validation_pool.select(
        range(min(len(train_subset), len(validation_pool)))
    )

    print(
        f"Using {len(train_subset)} training reviews and "
        f"{len(eval_subset)} validation reviews."
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=eval_subset,
        processing_class=TOKENIZER,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print("Trainer is set up. Ready to train the model.")
    prompt_user_to_continue()

    print("Step 6: Training the model...")
    trainer.train()
    print("Model training complete.")
    print("Evaluating the selected checkpoint on the official test split.")
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_imdb["test"],
        metric_key_prefix="test",
    )
    print(
        f"Final held-out test accuracy: "
        f"{test_metrics['test_accuracy']:.2%}"
    )
    prompt_user_to_continue()

    print("Step 7: Running inference on a sample text using the trained model.")
    text = "I absolutely loved this movie! The acting was phenomenal and the plot was engaging."
    print("Input text for sentiment analysis:")
    print(text)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=TOKENIZER)
    result = classifier(text)
    print("-" * 25, "Inference Result", "-" * 25)
    print(result)
    prompt_user_to_continue()

    print("Step 8: Interactive Inference")
    while True:
        print("Enter your own sentence for sentiment analysis. Type '-1' to exit.")
        user_input = input("Your sentence: ")
        if user_input.strip() == "-1":
            print("Exiting interactive mode.")
            break
        result = classifier(user_input)
        print("-" * 25, "Inference Result", "-" * 25)
        print(result)
        print("\n\n\n")


if __name__ == "__main__":
    main()
