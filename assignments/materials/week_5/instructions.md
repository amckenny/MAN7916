# Assignment 5: Unsupervised Machine Learning Approaches in Text Analysis

## Part 1 - Set up your Assignment 5 environment

1. Create a new folder in your `assignments/submissions` folder called `assignment_5`
2. In this folder, create new Python scripts (either normal Python(.py) or Jupyter(.ipynb) scripts) for the next two parts:
   1. `dictionary_embeddings_1.py`/`dictionary_embeddings_1.ipynb`
   2. `dictionary_embeddings_2.py`/`dictionary_embeddings_2.ipynb`
   3. `topic_models.py`/`topic_models.ipynb`

## Part 2 - Word embeddings and dictionary creation

Part of the dictionary creation process for dictionary-based computer-aided text analysis is the creation of initial word lists that can be evaluated for inclusion in the dictionary. Historically, this was done by evaluating every word in a corpus that appears at least three times in a text (to balance not overwhelming the raters with too many words that very infrequently appear in the texts and not missing words that are important but infrequent). In this assignment, we are going to see if we can improve on that. We are going to use the GloVe word embedding model to create an inductive list of words for potential inclusion in a dictionary. Our goal is going to be to create a comprehensive list of words that might be relevant but to omit words that are clearly not relevant.

Our dictionary is going to be of 'entrepreneurial' words. Your four root words are "entrepreneurial", "creative", "innovative", and "trailblazing". You believe the construct to be reflective of the the intersection of these four words rather than trying to capture the meaning of each root word individually. So, you are going to use the GloVe word embeddings to find words that are close to the intersection of these four words.

### With the first python file (`dictionary_embeddings_1.py`/`dictionary_embeddings_1.ipynb`), create a script that accomplishes the following:

1. Loads the GloVe word embedding (100d) model from the `local_data` directory
2. Identifies the word vectors for the four root words calculates the cosine similarity between each pair of root words and saves this in the `word_embedding_results.docx` file
3. One of the words should seem suspiciously low in similarity relative to the other three. Let's drop that word, maybe that wasn't a very good root word. Save the three remaining root words in a list.
4. Calculate the average vector of the three remaining root words and save the first five dimensions of this vector in the `word_embedding_results.docx` file
5. Calculate the cosine similarity between the average vector and each root word. Save this in the `word_embedding_results.docx` file.
6. Find the 50 words that are closest to the average vector in the GloVe model and save them in a list from most similar to least - you may want to explore the 'similar_by_vector' method in the gensim model. This will be our 'deductive' word list.
7. Open the `article_preprint.txt` from the `week_4` folder and find the 50 words *from that text* that are closest to the average vector. This will be our 'inductive' word list.
8. Combine the two word lists (inductive and deductive), remove any duplicates, and save them in a csv file called `word_list_for_evaluation.csv` with columns `word`, `score`, and `eval`. The `score` column should be the cosine similarity between the word and the average vector and the `eval` column should be empty.
9. End the program

Open the .csv file in Excel and evaluate each word for inclusion in the dictionary. Put a 1 in the `eval` column for the words to include in your dictionary and a 0 for the words to exclude. Save the file with your evaluations.

### With the first python file (`dictionary_embeddings_2.py`/`dictionary_embeddings_2.ipynb`), create a script that accomplishes the following:

1. Open the evaluated word list from the previous step and filter out the words that were not included in the dictionary.
2. Loads the GloVe word embedding (100d) model from the `local_data` directory
3. Use t-SNE to reduce the dimensionality of the word vectors to 2 dimensions
4. Plot the words in the 2D space and save the plot either in the Word document (`word_embeddings_results.docx`) or as `dictionary_embedding_plot.jpg` - make sure each word is labeled
5. End the program
6. At the bottom of the Word document, write a paragraph about the distribution of the words in the 2D space. Are there any clusters? Are there any words that are particularly far from the others? Are there any words that are particularly close to each other? What would be your assessment of the quality of the dictionary based on this visualization?

**The [Word embeddings](../../../tutorials/word_embeddings.ipynb) tutorial notebook should be helpful in completing this assignment**

**The use of generative AI to help you create/debug the Python code for this assignment *is* permitted**

<br>

## Part 3 - (Topic Modeling)

In the second part of the assignment, you will create a topic model of an area of research you are interested in. Collect a corpus of at least 10 articles on a phenomenon you are interested in. You may either use a PDF Python package to extract the text from the PDFs or copy and paste the text into .txt files, but do not upload the fulltexts to the github server (put them in the `local_data` folder).

1. Train LDA models with 2, 3, 4, 5, 6, 7, 8, 9, and 10 topics on the corpus (you may add more if you think there are more). Save the coherence score for each model in `topic_model_results.docx`.
2. Plot the coherence scores for each model (similar to [this](https://www.researchgate.net/publication/366941274/figure/fig1/AS:11431281111664783@1673099185937/Plot-of-topics-versus-coherence-score.png))and save the plot as `coherence_scores.jpg` - you do not have to use Python to do this, you can use Excel or another program if you prefer.
3. When you have the number of topics that you think best represents the corpus, tinker around with the alpha and beta hyperparameters to see if you can improve the coherence score. Save the coherence score for the best model in `topic_model_results.docx`. It's okay if you can't improve the coherence score.
4. Create a word cloud visualization for your best model's topics and save it as `topic_model_wordcloud.jpg` (or as individual files for each topic, if you prefer).
5. Identify one article that the model did not use in training and print the topics for that article. Save the topics in the `topic_model_results.docx` file. What do you think? Did the model do a good job of identifying the topics in the article?

**The [Topic modeling](../../../tutorials/topic_models.ipynb) tutorial notebook should be helpful in completing this assignment**

**The use of generative AI to help you create/debug the Python code for this assignment *is* permitted**

<br>

## Step 4 - Upload the assignment materials to the github server

1. Verify that the assignment_5 folder has been created and contains the three python files, and several output files.
2. Commit the new folder and files to your local repository
3. Push the new folder and files to the github server
4. Check that the file is now on the github server by visiting your repository on the github website

## Step 5 - Create an 'Issue' with a 'submitted' label

1. In the Issues tab create the ***submitted*** label
2. Create a new issue called ***Assignment 5 Submission***
    * with the submitted label
    * with the commit containing your assignment linked in the body
    * assigned to me (***amckenny***)
3. That's it!
