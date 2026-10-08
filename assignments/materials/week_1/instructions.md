# Assignment 1: Intro to Content Analysis and Tools

## Prerequisites:
 - Basic understanding of Git/GitHub
   - If you don't have this, I recommend completing the "GitHub Concepts" assignment in the [DataCamp course](https://app.datacamp.com/groups/man-7916-text-analysis-methods-9a2c4a02-a646-4664-b12f-83ae3ca6b350/dashboard) before proceeding
 - Basic understanding of Python
   - If you don't have this, I recommend completing the following assignments in the [DataCamp course](https://app.datacamp.com/groups/man-7916-text-analysis-methods-9a2c4a02-a646-4664-b12f-83ae3ca6b350/dashboard) before proceeding
     - Introduction to Python
     - Intermediate Python
     - Regular Expressions in Python: Basic concepts of string manipulation chapter
   - Not strictly required for *this* assignment, but will be necessary for future assignments and if you wait until then to learn it, you will be quickly overwhelmed

## Part 1 - Sign up for a GitHub account (if you don't already have one)

1. Navigate to [https://www.github.com](https://www.github.com).
2. Click the sign up link and follow the instructions to create a GitHub account
<br>

## Part 2 - Fork the MAN7916 GitHub repository and share it with me

1. Navigate to my [MAN 7916](https://www.github.com/amckenny/MAN7916) repository.
2. Follow the instructions [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo#forking-a-repository) to fork the repository into your GitHub account.
3. Follow the instructions [here](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-access-to-your-personal-repositories/inviting-collaborators-to-a-personal-repository#inviting-a-collaborator-to-a-personal-repository) to share the repository with me (use my UCF email address).
<br>

## Part 3 - Install Python and Visual Studio Code

*This is optional if you want to use a different code editor; however if you want my assistance, this is what I will be most helpful with.*

1. If you are on your personal computer...
   1. Install the latest version of Python 3.12 from [https://www.python.org/downloads/](https://www.python.org/downloads/)
       * Scroll down to "Active Python releases" and click on the link for the latest version of Python 3.12
       * The default settings are generally fine for the installation process
       * Remember where you saved Python on your hard drive, as you will need this information later
   2. Install the latest version of Visual Studio Code from [https://code.visualstudio.com/](https://code.visualstudio.com/)
       * The default settings are generally fine for the installation process
   3. Open Visual Studio Code and install the Python extension
       * Click on the Extensions icon in the left-hand sidebar
       * Search for "Python" in the search bar
       * Click the green "Install" button next to the "Python" extension
       * *Note: You may need to restart Visual Studio Code after installing the extension*
   1. Install Git from [git SCM](https://git-scm.com/downloads)
       * The default settings are generally fine for the installation process
       * I recommending choosing Visual Studio Code as your default editor during the installation process
2. If you are on your UCF computer
   1. Open the Software Center application
   2. Search for "Python" and install Python 3.12, if available, otherwise install 3.13 - do not install 3.14
   3. Search for "Visual Studio Code" and install it
   4. Search for "Git (x64)" and install it

## Part 4 - Clone your the MAN7916 GitHub repository to your local machine and create the Python Environment

1. Clone your new MAN7916 repository to your local machine using Visual Studio Code
   1. Click on the "View" menu in the top left corner of the application
   2. Click on "Command Palette..."
   3. Type "Git: Clone" into the search bar
   4. Click on the "Git: Clone" option that appears
   5. Paste the URL of your forked repository into the box that appears
        * You can find this URL by clicking on the green "Code" button on the GitHub website and copying the URL that appears
        * The URL should look something like `https://github.com/your_github_username/MAN7916.git`
   6. Choose a location on your hard drive to save the repository and remember where you saved it.
2. Open the repository folder you just created in Visual Studio Code
    1. Click on the "File" menu in the top left corner of the application
    2. Click on "Open Folder..."
    3. Navigate to the folder where you saved the repository on your hard drive
    4. Click the "Select Folder" button
3. Create a virtual Python environment for the repository
    1. Click on the "View" menu in the top left corner of the application
    2. Click on "Command Palette..."
    3. Type "Python: Select Interpreter" into the search bar
    4. Click on the "+ Create Virtual Environment..." option that appears
    5. Click on the "Venv" option that appears
    6. Click on the Python 3.12/3.13 interpreter that you installed in Part 3
    7. For the name of the virtual environment, ".venv" is fine
    8. When it asks you to install project dependencies, click on "Install project dependencies"
    9. Check the "requirements.txt" box that appears and press OK
    10. You'll see a progress bar in the bottom right corner of the application as the environment is created and the dependencies are installed. This may take a few minutes.
4. Create a new folder in the `assignments/submissions/` folder called ***assignment_1***
5. Place a new (empty, at the moment) Word Document in the newly created folder called ***assignment_1.docx***
<br><br>

## OPTIONAL - Upgrade your Python installation to include CUDA support for GPU acceleration
1. If you have a NVIDIA GPU, you can install the CUDA version of Python packages to enable GPU acceleration for certain libraries (e.g., PyTorch, SpaCy)
   1. Follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch with CUDA support
   2. Follow the instructions [here](https://www.spacy.io/usage) to install spaCy with CUDA support

## Part 5 - Run a Python Script and Jupyter Notebook

1. In Visual Studio Code, open the asst_1.py file in the `assignments/instructions/assignment_1/` folder
2. Run the script by clicking the play button in the top right corner of the application
3. This will run the python program and print the output to the terminal
    1. When it is running, click on the terminal at the bottom to interact with the program
    2. When it is done running, the output will be displayed in the terminal - take note of the text
    that is displayed. You will need this for your assignment.
    3. When done, click the trash can icon in the terminal to close the terminal
4. Open the asst_1.ipynb file in the `assignments/instructions/assignment_1/` folder
5. Click the "Run All" button in the top of the application (it looks like two overlapping play buttons)
    1. You may have to select a Python environment to do this - choose the one you created in Step 4
    2. This will run the Jupyter Notebook and print the output to the notebook
    3. When it is done running, the output will be displayed in the notebook - take note of the text


## Part 6 - Populate your assignment_1.docx file with the assignment contents

1. Open the ***assignment_1.docx*** file you created in Step 4 in Word
2. Give the document a normal assignment heading including
    * Your name
    * "MAN 7916 - Assignment 1"
3. Add a section header entitled "Python Script Output"
    * Paste the output from the Python script you ran in Step 5 into this section
    * Paste the output from the Jupyter Notebook you ran in Step 5 into this section as well
4. Add a section header entitled "Disease Stigma"
    * Visit: [This paper's GitHub repository](https://github.com/arsena-k/disease_stigma)
    * include in this section the following information:
        * On what date did the authors create the repository? [Hint](https://docs.github.com/en/pull-requests/committing-changes-to-your-project/creating-and-editing-commits/about-commits#using-the-file-tree)
        * On what date was the last commit entitled 'v1' made? [Hint](https://docs.github.com/en/pull-requests/committing-changes-to-your-project/creating-and-editing-commits/about-commits#using-the-file-tree)
        * What changed in the commit entitled 'fixed typo'? [Hint](https://docs.github.com/en/pull-requests/committing-changes-to-your-project/creating-and-editing-commits/about-commits#using-the-file-tree)
5. Add a section header entitled "Kiley's CARMA repository"
    * Visit: [Jason Kiley's CARMA repository](https://github.com/jtkiley/carma_python)
    * include in this section the following information:
        * How many 'Releases' are there? [Hint](https://docs.github.com/en/repositories/releasing-projects-on-github/viewing-your-repositorys-releases-and-tags)
        * What was the title of the main README.md file for the v1.0.0 release? [Hint](https://docs.github.com/en/repositories/releasing-projects-on-github/viewing-your-repositorys-releases-and-tags)
        * How many files changed between the v1.0.0 and v2.0.0 releases? [Hint](https://docs.github.com/en/repositories/releasing-projects-on-github/comparing-releases)
6. Save the file
<BR><BR>

## Part 7 - Upload the assignment to the GitHub server

1. Commit the new folder and file to your local repository
    * If using GitHub Desktop, you can do this by:
        * Making sure the new folder and file are checked in the "Changes" tab in the left sidebar of the application,
        * Typing `Added assignment_1.docx` in the "Summary" box in the bottom left corner of the application, then
        * clicking the "Commit to main" button in the bottom left corner of the application
    * If using git SCM, you can do this by:
        * Typing `git add .` in the terminal/command line interface and submitting that line (enter), then
        * Typing `git commit -m "Added assignment_1.docx"` in the terminal/command line interface and submitting that line (enter).
    * *Note: you can replace the message with whatever you want, but it's good practice to make it descriptive of what you did.*
2. Push the new folder and file to the GitHub server
    * If using GitHub Desktop, you can do this by clicking the "Push origin" button in the top right corner of the application
    * If using git SCM, you can do this by typing `git push` in the terminal/command line interface.
3. Check that the file is now on the GitHub server by visiting your repository on the GitHub website

## Part 8 - Create an 'Issue' with a 'submitted' label

[![Watch the video](https://img.youtube.com/vi/QqYxxX0nB6s/maxresdefault.jpg)](https://youtu.be/QqYxxX0nB6s)

*Note that I created this video for a different course, so the repository/file names are different. The steps are the same, though.*

1. In the Issues tab create the ***submitted*** label
2. Create a new issue called ***Assignment 1 Submission***
    * with the submitted label
    * with the commit containing your assignment linked in the body
    * assigned to me (***amckenny***)
3. That's it!
