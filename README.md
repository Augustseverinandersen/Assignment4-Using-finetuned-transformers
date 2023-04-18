[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/BhnScEmU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10838452&assignment_repo_type=AssignmentRepo)

# Assignment 4 - Using finetuned transformers via HuggingFace

# Contribution 

- The code used in this assignment is inspired from the class notebooks, and in collaboration with other students. All comments are written by me. 
- In this assignment I have used an emotion classifier made by Jochen Hartmann. Jochen Hartmann, "Emotion English DistilRoBERTa-base". https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/, 2022. The classifier has 7 catogires, anger, disgust, fear, joy, neutral, sadness, and surprise. It has been trained on 6 datasets, and is built on BERT. 
- The data used in this assignment is from Kaggle user Jillani Soft Tech: https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news The data is a csv file containing Fake and Real news. Their are four columns:
1. number: ID
2. title: The title of the news article
3. text: The text in the news article
4. label: The label is either fake or real. 

# Assignment descripition 

For this assignment, you should use ```HuggingFace``` to extract information from the *Fake or Real News* dataset that we've worked with previously.

You should write code and documentation which addresses the following tasks:

- Initalize a ```HuggingFace``` pipeline for emotion classification
- Perform emotion classification for every *headline* in the data
- Assuming the most likely prediction is the correct label, create tables and visualisations which show the following:
  - Distribution of emotions across all of the data
  - Distribution of emotions across *only* the real news
  - Distribution of emotions across *only* the fake news
- Comparing the results, discuss if there are any key differences between the two sets of headlines

# Methods / what the code does 

- The code loads the data in from a user specified path. Gets the model from huggingface by using pipeline, and specifying which model to get. The data is filtered and cleaned to generate a list of all headings, all real headings, and all fake headings. Thereafter, three loops are used to take each heading in the list, and use the emotion classifer to get the label (emotion) with the highest score, and store it in an empty list. Than using python code, each emotion occorance is counted, for all headings, real headings, and fake headings. After some data rangling, splitting dictionary key and value pairs, the data is visualized into four visualizations, created with matplotlib.

# Discussion

## Visualization One

## Visualization Two

## Visualization Three

## Visualization Four

# Usage

To use the script I have created follow these steps, after cloning the repository:

1. Get the data from Kaggle: https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news and store it in the data folder.
2. Run bash setup.sh in the command line, which will create a virutal environment, and install the nessecary requirements. 
3. OBS! Write the filepath to where you have stored the data, if you haven't stored it in the data folder: In the commandline run python3 src/emotions.py --filepath ./data/fake_or_real.csv
4. After the code has run, the vizualizations will be stored in the folder figs.



## Tips
- I recommend using ```j-hartmann/emotion-english-distilroberta-base``` like we used in class.
- Spend some time thinking about how best to present you results, and how to make your visualisations appealing and readable.
- **MAKE SURE TO UPDATE YOUR README APPROPRIATELY!**
