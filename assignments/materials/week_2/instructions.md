# Assignment 2: Corpus Collection

## Prerequisites:
 - Basic understanding of scraping in Python
   - If you don't have this after completing the tutorial, I recommend completing the "Intermediate Importing Data in Python: Importing data from the Internet chapter" assignment in the [DataCamp course](https://app.datacamp.com/groups/man-7916-text-analysis-methods-9a2c4a02-a646-4664-b12f-83ae3ca6b350/dashboard) before proceeding.
 - Basic understanding of API access with Python
   - If you don't have this after completing the tutorial, I recommend completing the "Intermediate Importing Data in Python: Interacting with APIs to Import data from the web chapter" assignment in the [DataCamp course](https://app.datacamp.com/groups/man-7916-text-analysis-methods-9a2c4a02-a646-4664-b12f-83ae3ca6b350/dashboard) before proceeding.

## Part 1 - Set up your Assignment 2 environment

1. Create a new folder in your `assignments/submissions` folder called `assignment_2`
2. In this folder, create new Python scripts (either normal Python(.py) or Jupyter(.ipynb) scripts) for the next two parts:
   1. `dean_scraping.py`/`dean_scraping.ipynb`
   2. `ten_years_api.py`/`ten_years_api.ipynb`

## Part 2 - What is the Dean thinking?

You want to stay abreast of what Dean Jarley is thinking, but you don't have time to read all of his blog posts. You decide to scrape the [dean's blog](https://pauljarley.wordpress.com/) to get the text of his posts. Don't worry about getting *all* of the posts, just get the first page of posts.

With the first python file, create a script that:

1. Visits the landing page for the dean's blog and scrapes the following information:
    * The title of the post
    * The date of the post
    * The url of the post
2. Visits each of the urls from the previous step and scrapes the following information:
    * The text of the post
    * The comments
      * The author of the comment
      * The text of the comment
3. Saves the information to a CSV file titled "dean_scraping.csv" in the assignment_2 folder

The CSV file should have the following columns and look something like:

| title | url | entry_date | post_text | comments |
| --- | --- | --- | --- | --- |
| We Should All Toss Something into the Tip Jar | https://pauljarley.wordpress.com/2024/10/21/we-should-all-toss-something-into-the-tip-jar/ | 2024-10-21T03:04:00-04:00 | "This is the text of the post" | Commenter 1: Their Comment \n Commenter 2: Their Comment |

**The [ucf_news_scraping.ipynb](../../../tutorials/ucf_news_scraping.ipynb) tutorial notebook should be helpful in completing this assignment**

**The use of generative AI to help you create/debug the Python code for this assignment *is* permitted**

<br>

## Step 3 - Ten Years of Subfield Journal Publications

With the second python file, create a script that accomplishes the following:

1. Choose three journals in your subfield of interest (e.g., OB, Strategy, Entrepreneurship)
2. Use the CrossRef API to get a list every article they have published since Jan. 1, 2015. Retrieve the following fields:
   1. DOI
   2. Title
   3. Number of references
   4. Number of authors
   5. Number of citations
   6. URL
   7. Journal
   8. Abstract (where present)
   9. Publication year
3. Save the information to a CSV file titled "ten_years_api.csv" in the assignment_2 folder
4. Use this dataset to:
   1. Calculate the correlation between publication year and citation count
   2. Calculate an ANOVA showing whether there are mean differences by journal.
   3. Save the results of the correlation and ANOVA analyses to "publication_stats.txt" in the assignment_2 folder

The CSV file should have the following columns and look something like:

| doi | title | n_refs | n_authors | n_cites | url | journal | abstract | pub_year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10.1016/j.jbusvent.2021.106175 | Failed but validated? The effect of market validation on persistence and performance after a crowdfunding failure | 152 | 3 | 17 | http://dx.doi.org/10.1016/j.jbusvent.2021.106175 | Journal of Business Venturing | (abstract text) | 2022 |

**The [crossref_api.ipynb](../../../tutorials/crossref_api.ipynb) tutorial notebook should be helpful in completing this assignment**

**The use of generative AI to help you create/debug the Python code for this assignment *is* permitted**

<br>

## Step 4 - Upload the assignment materials to the github server

1. Verify that the assignment_2 folder has been created and contains the two python files, two CSV files, and one text file.
2. Commit the new folder and files to your local repository
3. Push the new folder and files to the github server
4. Check that the file is now on the github server by visiting your repository on the github website

## Step 5 - Create an 'Issue' with a 'submitted' label

1. In the Issues tab create the ***submitted*** label
2. Create a new issue called ***Assignment 2 Submission***
    * with the submitted label
    * with the commit containing your assignment linked in the body
    * assigned to me (***amckenny***)
3. That's it!
