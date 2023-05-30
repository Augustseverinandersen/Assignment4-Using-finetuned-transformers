# 4. Assignment 4 - Using finetuned transformers via HuggingFace
## 4.1 Assignment description
Written by Ross: 

For this assignment, you should use _HuggingFace_ to extract information from the _Fake or Real News dataset_ that we've worked with previously. You should write code and documentation which addresses the following tasks:
-	Initialize a _HuggingFace_ pipeline for emotion classification
-	Perform emotion classification for every headline in the data.
-	Assuming the most likely prediction is the correct label, create tables and visualisations which show the following:
    - Distribution of emotions across all the data
    - Distribution of emotions across only the real news
    - Distribution of emotions across only the fake news
-	Comparing the results, discuss if there are any key differences between the two sets of headlines.
## 4.2 Machine Specification and My Usage
All the computation done for this project was performed on the UCloud interactive HPC system, which is managed by the eScience Center at the University of Southern Denmark. The scripts were created with Coder Python 1.73.1 and Python version 3.9.2. This script was run on 16 CPUs, and a total run time of 25 minutes.
### 4.2.1 Perquisites
To run the scripts, make sure to have Bash and Python3 installed on your device. The script has only been tested on Ucloud.
## 4.3 Contribution
The code used in this assignment is created in collaboration with other students. All comments are written by me. 

In this assignment, I have used an emotion classifier made by Jochen Hartmann. Jochen Hartmann, "Emotion English DistilRoBERTa-base". https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/, 2022. The pre-trained model is used to classify six emotions and a neutral category. The model was trained on six different datasets ranging from _Reddit to student self-reports_ and achieved an accuracy of 66% for predicting the right emotion. Furthermore, the model is built on BERT architecture. 

The data used in this assignment is from Kaggle user [Jillani Soft Tech](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news). 
### 4.3.1 Data
The data used in this assignment is the same as in _Assignment-2_. The data is a CSV file containing _fake_ and _real_ news. There are four columns (_number, title, text, and label_) and over 6,000 rows. In this assignment I am using the column _title_, to get the emotion from each headline. 
## 4.4 Packages
-	**Matplotlib (version 3.7.1)** is used to create the bar and scatterplots.
-	**Pandas (version 2.0.1)** is used to read the data, and data manipulation.
-	**Transformers (version 4.28.1)** is used to import _pipeline_ which is used to access the model from _HugginFace_.
-	**TensorFlow (version 2.11.1)** is used to enable downloading the transformers packages.
-	**Zipfile** is used to unpack the zip file.
-	**Sys** is used to navigate the directory.
-	**Argparse** is used to create command-line arguments.
-	**Os** is used to navigate file paths across operating systems.

## 4.5 Repository Contents 
The repository contains the following folders and files:
-	***Data*** is an empty folder where the zip file will be placed.
-	***Figs*** is the folder that contains the bar and scatter plots created in the script.
-	***Out*** is the folder that contains the tables created in the script.
-	***Src*** is the folder that contains the script ``emotions.py``.
-	***README.md*** is the README file.
-	***Requirements.txt*** is the file that contains the packages that must be installed to run the script.
-	***Setup.sh*** is the file that creates a virtual environment, upgrades pip, and installs the package from requirements.txt.
## 4.6 Methods
-	The script starts by initializing an argparse that is used to get the path to the zip file. 
-	The zip file is then unpacked, and the CSV file is read as a Pandas data frame. 
-	The pre-trained model is then loaded in using _pipeline_ and stored in the variable _classifier_. 
-	The Pandas data frame is then filtered to create two more data frames. One with only _FAKE_ news and one with only _REAL_ news. 
-	Then, the column text for each data frame is stored as a Pandas series in a variable. This is done so I can iterate over each headline and get the emotion.
-	After creating the Pandas series, a for loop loops over each headline, using the classifier to get the emotion with the highest score. The emotion is stored in a list. When the loop is done, a new loop starts that counts each occurrence of each emotion and stores it in a dictionary. The _keys_ are the emotions, and the _values_ are the count of that emotion. The dictionary is then sorted alphabetically by the _keys_. 
-	The dictionaries are then split into lists. One list stores the _keys_, and the other stores the _values_.
-	A table is then created for _all headlines, real headlines, and fake headlines_. The table has two columns, _emotion_ and _count_. The tables are saved to the folder _out_. 
-	The bar graphs are then created with the x-axis for emotions and the y-axis for the count. A bar graph is created for _all headlines_, _real headlines_, and _fake headlines_ and saved in the folder _figs_.
-	Lastly, a scatterplot is created. This scatterplot has unique markers for which of the three headlines it is displaying. The scatterplot is saved in the folder _figs_.
## 4.7 Discussion
The pre-trained model used in this assignment is good for doing sentiment analysis. In my case, sentiment analysis of news headlines. With the seven categories available for each headline, it is interesting to see which emotion is the most represented of all the news headlines. 

**All Emotions Bar**

This bar plot gives an overview of the general distribution of emotions for all headings. A vast majority of the headings are classified as neutral (3180), followed by fear (1076), anger (795), sadness (487), disgust (434), surprise (208), and then joy (155).

**Real Emotions Bar**

This bar plot shows the distribution of emotions for real news articles. Again, the vast majority are categorised as neutral (1649), followed by fear (555), anger (383), sadness (245), disgust (186), surprise (90), and then joy (63).

**Fake Emotions Bar**

This bar plot shows the distribution of emotions for all the fake news articles. The majority are categorised as neutral (1531), followed by fear (521), anger (412), disgust (248), sadness (242), surprise (118), and then joy (92).

**All, Real, and Fake Emotions Scatterplot**

This scatterplot shows all three categories _(all, real, fake)_ in one plot. This plot displays how common the emotion label is, for each category. It also shows how real and fake headlines, nearly are identical for each emotion.

**Findings**

By using the plots, I can clearly state the majority of news headlines are classified as neutral. This could lead to the assumption that the headlines are mostly factual and that the author is not trying to use emotion to try to sway the reader in a certain direction. Removing the neutral category, it can be stated that negative emotions, such as fear and anger, are more dominant in news headlines. Lastly, it can be deduced that there are no major differences between the REAL headlines and the FAKE headlines when looking at emotions.

## 4.8 Usage
Follow these steps to run the script:
-	Clone the repository.
-	Navigate to the correct directory.
-	Get the zip file from [here](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news), and place it in the data folder (You might need to rename the zip file).
-	Run ``bash setup.sh`` in the command line, this will create a virtual environment and install the requirements.
    - **OBS!** You need to accept an installation of some packages.
-	Run ``source ./assignment_4/bin/activate`` in the command line to activate the virtual environment.
-	Run ``python3 src/emotions.py --zip_path data/archive.zip`` in the command line to run the script.
    - The argparse ``--zip_path`` takes a string as input and is the path to your zip file.
-	The tables created will be stored in the folder out and the visualizations will be stored in the folder _figs_.
