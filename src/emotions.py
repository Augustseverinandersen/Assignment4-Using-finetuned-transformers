# Huggingface pipeline
from transformers import pipeline 

# Data munging tools 
import pandas as pd 
import matplotlib.pyplot as plt
import zipfile

# System tools
import sys
sys.path.append(".")
import argparse
import os 


def input_parse():
    # initialize the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=str, help = "The path to the zip folder") # argument is zippath as a string
    args = parser.parse_args()

    return args


def unzip(args):
    folder_path = os.path.join("data", "fake_or_real_news.csv") # Defining the folder_path to the data.
    if not os.path.exists(folder_path): # If the folder path does not exist, unzip the folder, if it exists do nothing 
        print("Unzipping file")
        path_to_zip = args.zip_path # Defining the path to the zip file
        zip_destination = os.path.join("data") # defining the output destination

        with zipfile.ZipFile(path_to_zip,"r") as zip_ref: # using the package from zipfile, to un zip the zip file
            zip_ref.extractall(zip_destination) # Unzipping
    print("The files are unzipped")
    return folder_path


def loading_data(folder_path):
    print("Loading data")
    data = pd.read_csv(folder_path) # Loading the data into a pandas dataframe
    return data


def pre_trained_model():
    print("Getting model")
    classifier = pipeline("text-classification", # I am using huggingface's pipelin to load the text classifier
                      model="j-hartmann/emotion-english-distilroberta-base", # The pretrained model
                      return_all_scores=False) # Only return the emotion with the highest score 
    return classifier


def filtering(data):
    print("Creating REAL or FAKE dataframes")
    data_filtered_real = data[data['label'] == 'REAL'] # Creating a dataframe of only REAL news
    data_filtered_fake = data[data['label'] == 'FAKE'] # Creating a dataframe of only FAKE news
    return data_filtered_real, data_filtered_fake


def data_cleaning(data, data_filtered_real, data_filtered_fake):
    print("Selecting column")
    all_texts = data["title"] # Only selecting column "title"
    real_texts = data_filtered_real["title"] # Only selecting column "title"
    fake_texts = data_filtered_fake["title"] # Only selecting column "title"
    
    return all_texts, real_texts, fake_texts



def emotion_loop(classifier, texts):
    print("Finding emotions and counting...")
    emotion_list = [] # Creating an empty list to store each emotion label.
    emotion_count = {}
    for line in texts: # For loop that goes through each headline in the pandas series, and uses the classifier.
        emotion_list.append(classifier(line)[0]["label"])  # append list #output[0]["label"]
        # using classifier to get the emotion of each headline. The emotions are stored as Counter object dictionaries in a list. [0] gets the first dictionary in the list ["label"].
        # by only selecting label, I append it to the empty list emotion_list.
    
    for emotion in emotion_list: # For every emotion in the list 
        if emotion in emotion_count: # If the emotion appears 
            emotion_count[emotion] += 1 # add one to the value of the correct key in the empty dictionary 
        else: # Else statement adds the emotion to the dictionary is it has not appeared 
            emotion_count[emotion] = 1 # and adds one to the value
    emotion_count = dict(sorted(emotion_count.items())) # Ordering the keys alphabetically 
    return emotion_count


def splitting_dictionary(emotion_count, real_emotion_count, fake_emotion_count):
    print("Preparing for visualization")
    all_emotions = list(emotion_count.keys()) # Converting the keys to a list 
    all_count = list(emotion_count.values()) # Converting the values to a list 

    real_emotions = list(real_emotion_count.keys())
    real_count = list(real_emotion_count.values())

    fake_emotions = list(fake_emotion_count.keys())
    fake_count = list(fake_emotion_count.values())

    return all_emotions, all_count, real_emotions, real_count, fake_emotions, fake_count


def tabels(emotions, tablename):
    print("Creating and saving tables")
    table = pd.DataFrame(list(emotions.items()), columns=["Emotion", "Count"]) # Creating a DataFrame with the emotions and counts as columns
    output_path = os.path.join("out", tablename + ".csv")
    table.to_csv(output_path, index=False) # Save the DataFrame to a CSV file without including the index


def visualization_bar(emotions, count, title, folder = "figs"): # Creating a bar graph to visualise the emotions. Can be used for all, real, or fake
    print("Creating bar plots")
    plt.bar(emotions, count)  # X axis is emotions, Y axis is count 
    plt.xlabel("Emotions") # X axis label
    plt.ylabel("Count") # Y axis label
    plt.title(title) # Giving it a title

    filename = os.path.join(folder, title.replace(' ', '_') + '.png') # Saving the plot. Remove space with underscore in the title
    plt.savefig(filename)
    plt.clf() # Clearing the plot


def visualization_scatter(all_emotions, all_count, real_emotions, real_count, fake_emotions, fake_count, folder = "figs"):
    print("Visualizing scatterplot")
    # Plotting all, real, and fake in one scatter plot
    plt.scatter(all_emotions, all_count, c='blue', label='All Headings', marker= "o") # Creating all headings points as a blue circle
    plt.scatter(real_emotions, real_count, c='green', label='Real Headings', marker= "+" ) # Creating real headings as a green plus sign
    plt.scatter(fake_emotions, fake_count, c='red', label='Fake Headings', marker= "_") # Creating fake headings as a red underscore

    plt.xlabel('Emotion') # X axis label
    plt.ylabel('Count') # Y axis label
    plt.legend() # Creating a legend
    plt.title('Emotion Counts') # Title

    filename = os.path.join(folder, "scatterplot" + '.png') # Saving the plot
    plt.savefig(filename)
    plt.clf() # Clearing plot


def main_function():
    args = input_parse() # Command line arguments 
    folder_path = unzip(args) # Unzipping zip folder
    data = loading_data(folder_path) # Loading data into pandas dataframe
    classifier = pre_trained_model() # Getting pretrained model
    data_filtered_real, data_filtered_fake = filtering(data) # Creating two new dataframes with FAKE or REAL
    all_texts, real_texts, fake_texts = data_cleaning(data, data_filtered_real, data_filtered_fake) # Selecting column
    emotion_count = emotion_loop(classifier, all_texts)# Getting the emotion for each headline in all headlines
    real_emotion_count = emotion_loop(classifier, real_texts) # Getting the emotion for each headline in REAL headlines
    fake_emotion_count = emotion_loop(classifier, fake_texts)# Getting the emotion for each headline in FAKE headlines
    all_emotions, all_count, real_emotions, real_count, fake_emotions, fake_count = splitting_dictionary(emotion_count, real_emotion_count, fake_emotion_count) # Splitting the values
    tabels(emotion_count, "All_headings") # Creating and saving table for all headings
    tabels(real_emotion_count, "Feal_headings") # Creating and saving table for REAL headings
    tabels(fake_emotion_count, "Fake_headings") # Creating and saving table for FAKE headings
    visualization_bar(all_emotions, all_count, "All Headings", folder = "figs") # Creating bar plot for all headings
    visualization_bar(real_emotions, real_count, "Real Headings", folder = "figs") # Creating bar plot for REAL headings
    visualization_bar(fake_emotions, fake_count, "Fake Headings", folder = "figs") # Creating bar plot for FAKE headings
    visualization_scatter(all_emotions, all_count, real_emotions, real_count, fake_emotions, fake_count, folder = "figs") # Creating a scatter plot

if __name__ == "__main__": # If script is called from command line fun main function
    main_function()
