[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/BhnScEmU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10838452&assignment_repo_type=AssignmentRepo)

# Assignment 4 - Using finetuned transformers via HuggingFace

## Contribution 

- The code used in this assignment is inspired from the in-class notebooks, and in collaboration with other students. All comments are written by me. 
- In this assignment I have used an emotion classifier made by Jochen Hartmann. Jochen Hartmann, "Emotion English DistilRoBERTa-base". https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/, 2022. The classifier has 7 catogires, anger, disgust, fear, joy, neutral, sadness, and surprise. It has been trained on 6 datasets, and is built on BERT. 
- The data used in this assignment is from Kaggle user [Jillani Soft Tech:](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news) The data is a csv file containing Fake and Real news. Their are four columns:
1. number: ID
2. title: The title of the news article
3. text: The text in the news article
4. label: The label is either fake or real. 

## Packages 

## Assignment descripition 
Written by Ross:
For this assignment, you should use ```HuggingFace``` to extract information from the *Fake or Real News* dataset that we've worked with previously.

You should write code and documentation which addresses the following tasks:

- Initalize a ```HuggingFace``` pipeline for emotion classification
- Perform emotion classification for every *headline* in the data
- Assuming the most likely prediction is the correct label, create tables and visualisations which show the following:
  - Distribution of emotions across all of the data
  - Distribution of emotions across *only* the real news
  - Distribution of emotions across *only* the fake news
- Comparing the results, discuss if there are any key differences between the two sets of headlines

## Methods / what the code does 

- The code loads the data in from a user specified path. Gets the model from huggingface by using pipeline, and specifying which model to get. The data is filtered and cleaned to generate a list of all headings, all real headings, and all fake headings. Thereafter, three loops are used to take each heading in the list, and use the emotion classifer to get the label (emotion) with the highest score, and store it in an empty list. Than using python code, each emotion occorance is counted, for all headings, real headings, and fake headings. After some data rangling, splitting dictionary key and value pairs, the data is visualized into four visualizations, created with matplotlib.

## Discussion

### All Emotions Bar
- This bar plot gives an overview of the general distripution of emotions for all headings. A vast majority of the headings are classified as neutral, followed by fear, anger, sadness, disgust, surprise, and than joy. 
### Real Emotions Bar
-  This bar plot shows the distripution of emotions for real news articles. Again, the vast majority are categorised as neutral, followed by the same ranking of emotions as the previous plot. 
### Fake Emotions Bar
- This bar plot shows the distripution of emotions for all the fake news articles. The majority are categorised as neutral, but here disgust is slightly bigger than, sadness. 
### All, Real, Fake Emotions Scatterplot
- This scatterplot shows all three categorises (all, real, fake) in one plot. This plot clearly displays how common the emotion label, is for each category. It also shows how real and fake headlines, nearly are identical for each emotion.

### Findings 
- By using the plots I can clearly state the the majority of emotions are classified as netural. This could lead to the assumption that the headlines are mostly factual, and the author not using emotion to try to sway the reader in a certain direction. Another clearly statement, which can be made, is that mostly negative emotions are used in news headlines, such as fear and anger. This portays nicely to the "real world", where most news articles are of murders or problems in society. 

### Future analysis 
- It could be interesting to use POS, to see which kinds of words are used in the headline that give an emotion. By doing so, you could also see which emotion charged word is used the most.

## Usage

To use the script I have created follow these steps, after cloning the repository:

1. Get the data from Kaggle: https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news and store it in a folder you have created, f.x. data
2. Run bash setup.sh in the command line, which will create a virutal environment, and install the nessecary requirements. 
3. Run source ./assignment_4/bin/activate  in the command line to activate the virutal environment
4. OBS! Write the filepath to where you have stored the data, if you haven't stored it in the data folder: In the commandline run python3 src/emotions.py --filepath ./data/fake_or_real_news.csv
5. After the code has run, the vizualizations will be stored in the folder figs.
