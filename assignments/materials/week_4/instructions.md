# Assignment 4: Introduction to Computer-Aided Text Analysis

## Prerequisites:
 - Basic understanding of plotting with mathplotlib in Python
   - If you don't have this after completing the tutorial, I recommend completing the "Intermediate Python: Matplotlib chapter" assignment in the [DataCamp course](https://app.datacamp.com/groups/man-7916-text-analysis-methods-9a2c4a02-a646-4664-b12f-83ae3ca6b350/dashboard) before proceeding.
 - Basic understanding of regular expressions in Python
   - If you don't have this after completing the tutorial, I recommend completing the following assignments in the [DataCamp course](https://app.datacamp.com/groups/man-7916-text-analysis-methods-9a2c4a02-a646-4664-b12f-83ae3ca6b350/dashboard) before proceeding.
     - "Regular Expressions in Python: Regular Expressions for Pattern Matching chapter"
     - "Regular Expressions in Python: Advanced Regular Expression Concepts chapter"

## Part 1 - Set up your Assignment 4 environment

1. Create a new folder in your `assignments/submissions` folder called `assignment_4`
2. In this folder, create new Python scripts (either normal Python(.py) or Jupyter(.ipynb) scripts) for the next two parts:
   1. `article_regexes.py`/`article_regexes.ipynb`
   2. `aussie_cata.py`/`aussie_cata.ipynb`

## Part 2 - References/Citations Check

Part of preparing a manuscript for submission to a journal is ensuring that all references are cited in the text and that all citations in the text have a corresponding reference. This can be a time-consuming process, especially for longer manuscripts. In this assignment, you will create a Python script that will help speed up this process.

With the first python file, create a script that:

1. Reads the file `article_preprint.txt` in the `materials/week_4` folder
2. Extracts all citations from the text. There are two types of citations in the text:
   1. Citations with names and dates in the parentheses (e.g., "(e.g., Smith et al., 2022; Jones, 2021; Johnson & Johnson, 2020)")
   2. Citations with only the dates in the parentheses (e.g., "Donaldson et al. (2022)")
3. Generates a bar chart of the most cited citations (i.e., the citations that appear the most frequently in the text)
   - Do this only for the citations with names and dates in the parentheses, no need to do fancy footwork to accommodate the other type of citation (which is less common and significantly more difficult to parse with the tools we have learned so far).
   - Save this file as `common_citations.jpg` in the `assignment_4` folder
   - See the [sample figure](./sample_comm_cites.jpg) for an example of what this file should look like
4. Generates 2 lists of all unique citations to look for in the references section
   - Save these lists in a text file called `references_report.txt` in the `assignment_4` folder
   - See the [sample report](./sample_refs_report.txt) for an example of what this file should look like

**Important Guidelines:**
- You must use regular expressions for all extractions/changes you are making to the text. While some of the things you will want to do are possible with string methods (e.g., splitting, replacing, slicing), you must use regular expressions for this assignment.
- For the bar chart,
  - You can use any Python library you like (e.g., matplotlib, seaborn, plotly).
  - The bar chart should have the citation names on the x-axis and the number of times they appear in the text on the y-axis.
  - The bar chart should be organized with the most commonly cited citation on the left and be in descending order of frequency.
  - The top 20 citations should be included in the bar chart.
- For the references report
  - Store the citations with names and dates in the parentheses in a separate list from the citations with only the dates in the parentheses
  - There should be one row per reference to look for in the references section - if an article is cited multiple times, it should only appear once in the report.
  - The report should be in alphabetical order by the first author's last name.

**Helpful Hints:**
- For Citations with names and dates in the parentheses:
  - It may be helpful to first find all the sets of citations in the text (such as "(e.g., Smith et al., 2022; Jones, 2021; Johnson & Johnson, 2020)") before then extracting the individual citations from each set (such as ["Smith et al., 2022", "Jones, 2021", "Johnson & Johnson, 2020"]).
  - Remember that prefixes such as "e.g.," and page references such as ", p. 123" are not part of the citations.
- For Citations with only the dates in the parentheses:
  - Don't worry about getting *just* the names with the citation dates - that will be difficult to do with the tools we have learned so far. Instead, try one of the following:
    - Easier: Extract the date with 50 characters before it. This will likely include the author's names as well as some other text, but it will be good enough for this assignment.
    - For a challenge: Extract the full sentence up to the date. This will require a more complex regular expression, but it is possible with the tools we have learned so far. This is what you see in the sample report.

**The [email_regex.ipynb](../../../tutorials/email_regex.ipynb) and [text_prep_and_dict.ipynb](../../../tutorials/text_prep_and_dict.ipynb) tutorial notebooks should both helpful in completing this assignment**

**The use of generative AI to help you create/debug the Python code for this assignment *is* permitted**

<br>

## Part 3 - Preprocessing and Dictionary-Based Analysis

In the second part of the assignment, you will use the preprocessing techniques discussed to prepare a corpus of articles for analysis, create a dictionary of terms, and analyze the articles using the dictionary. With the second python file, create a script that accomplishes the following:

1. In the `local_data/aussie/About` folder, there are a bunch of .txt files that contain the "About Us" sections of various Australian companies' websites from a *Family Business Review* article I published a long time ago. Load those into a pandas DataFrame.
   - 'name' column: the name of the file (e.g., 1, 2, 3, 10, etc.)
   - 'filepath' column: the path to the file (e.g., 'local_data/aussie/About/1.txt')
   - 'text' column: the text of the file
2. Use SpaCy to preprocess the text in the 'text' column twice:
   - The first time (column 'preprocessed_ws') should not remove any stopwords
   - The second time (column 'preprocessed_wos') should remove all stopwords
3. Create Counters of the word frequencies for both the 'preprocessed_ws' and 'preprocessed_wos' columns and generate line charts of the 100 most common words in each column.
   - Save these files as `common100_ws.jpg` and `common100_wos.jpg` in the `assignment_4` folder
   - See the [sample figure 1](./sample_common100_ws.jpg) and [sample figure 2](./sample_common100_wos.jpg) for an example of what these files should look like
4. (not a Python step) Navigate to the [CAT Scanner dictionaries website](https://www.catscanner.net/dictionaries/) and obtain the dictionary for the 'Innovativeness' dictionary associated with the McKenny, Aguinis, et al. (2018) article associated with Entrepreneurial Orientation.
   - You don't need the entire file, just the words - store them in a list or dict in your script.
5. Filter your preprocessed with stops Counter for only the words in the 'Innovativeness' dictionary and generate a line chart of the 20 most common words in the filtered Counter (there won't be 20 words in the results, but that's okay)
   - Save this file as `innov_freqs.jpg` in the `assignment_4` folder
   - See the [sample figure](./sample_innov_freqs.jpg) for an example of what this file should look like.
6. Create two final columns in the dataset:
   1. 'innov_ws': the frequency of the words in the 'Innovativeness' dictionary in the 'preprocessed_ws' column
   2. 'innov_perwd_ws': the frequency of the words in the 'Innovativeness' dictionary in the 'preprocessed_ws' column divided by the total number of words in the 'preprocessed_ws' column.
7. Save the dataset as a CSV file called `innov_aussie_data.csv` in the `assignment_4` folder
   - See the [sample CSV](./sample_innov_data.csv) for an example of what this file should look like.

**The [text_prep_and_dict.ipynb](../../../tutorials/text_prep_and_dict.ipynb) tutorial notebook should be helpful in completing this assignment**

**The use of generative AI to help you create/debug the Python code for this assignment *is* permitted**

<br>

## Step 4 - Upload the assignment materials to the GitHub server

1. Verify that the assignment_4 folder has been created and contains the two python files, four jpg files, two csv files, and one txt file.
2. Commit the new folder and files to your local repository
3. Push the new folder and files to the GitHub server
4. Check that the file is now on the GitHub server by visiting your repository on the GitHub website

## Step 5 - Create an 'Issue' with a 'submitted' label

1. In the Issues tab create the ***submitted*** label
2. Create a new issue called ***Assignment 4 Submission***
    * with the submitted label
    * with the commit containing your assignment linked in the body
    * assigned to me (***amckenny***)
3. That's it!
