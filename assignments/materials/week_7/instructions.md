# Assignment 7: Summative Text Analysis Assignment

## Part 1 - Set up your Assignment 7 environment

1. Create a new folder in your `assignments/submissions` folder called `assignment_7`
2. In this folder, create a new Python script (either normal Python(.py) or Jupyter(.ipynb) script) for the assignment: `summative_assignment.py`/`summative_assignment.ipynb`. If you would like to break this into multiple files, you can do so, but if you'd like to complete the assignment in one file, that is fine too.

## Part 2 - Summative Text Analysis Assignment Instructions

In this assignment you are going to pull together a lot of what you have learned in various weeks of the course. You are going to train a supervised machine learning model to classify abstracts into one of two categories. You will then evaluate the performance of your model.

1. Choose two journals associated with a single subdiscipline (e.g., entrepreneurship, strategy, international business, organizational behavior, etc) and use the Crossref API to download the abstracts of all articles published since 2020 in these journals. Make sure the journals you collect actually have abstracts in the Crossref database... not all do. You should be familiar with how to do this from the Week 2 assignment.
2. Use either regular expressions or a large language model to remove copyright statements from the abstracts. You should be familiar with how to do this from the week 4 assignment or week 7 materials.
3. Obtain the embedding vectors for the abstracts using two different approaches: (1) tf-idf and (2) openAI embeddings. You should be familiar with how to do this from the week 5 assignment.
4. Train supervised machine learning models to classify the abstracts into one of the two categories.
    * You can use any supervised machine learning model you would like, but you should try at least two different models.
    * Make sure to use cross-validation and Grid Search/Randomized Search to optimize the hyperparameters of your models.
    * Make sure to compare the performance of tf-idf and openAI embedding-based models.
    * You should be familiar with how to do this from this week's tutorials.
5. Obtain 20 abstracts from the Journal of Management that the model was not trained on. Manually code these as to whether they are in the category of the first journal or the second journal.
    * Use good manual coding practices from week 3, but don't worry about choosing the *ideal* coders, use eachother as coders for efficiency.
    * Use your model to predict the category of these abstracts and compare the predicted categories to your manual coding.
6. Write up your results in a text file called `results.docx` in your assignment_7 folder. You should include the following in your write up:
    * The names of the two journals you chose and the number of abstracts you downloaded from each.
    * A description of how you removed copyright statements from the abstracts.
    * A description of how you obtained the embedding vectors for the abstracts.
    * A description of the supervised machine learning models you trained, including:
        * The model names
        * The performance metrics for each model (e.g., accuracy, precision, recall, F1 score, MCC)
        * A comparison of the performance of tf-idf and openAI embedding-based models
        * Identify the best performing model
    * A description of the 20 abstracts you manually coded from the Journal of Management, including:
        * A confusion matrix comparing the manual coding to the model predictions
        * Your assessment of the model's performance based on the confusion matrix
        * Any other interesting observations from your analysis

**All tutorial notebooks from this semester should be helpful in completing this assignment**

**The use of generative AI to help you create/debug the Python code for this assignment *is* permitted**

<br>

## Part 3 - Upload the assignment materials to the github server

1. Verify that the assignment_7 folder has been created and contains at least the python and Word files.
2. Commit the new folder and files to your local repository
3. Push the new folder and files to the github server
4. Check that the file is now on the github server by visiting your repository on the github website

## Part 4 - Create an 'Issue' with a 'submitted' label

1. In the Issues tab create the ***submitted*** label
2. Create a new issue called ***Assignment 7 Submission***
    * with the submitted label
    * with the commit containing your assignment linked in the body
    * assigned to me (***amckenny***)
3. That's it!
