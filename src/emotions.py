from transformers import pipeline 
import pandas as pd 
import os 
from collections import Counter
import matplotlib.pyplot as plt

import argparse
import sys
sys.path.append(".")




def input_parse():
    # initialize the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str) # argument is filepath as a string
    args = parser.parse_args()

    return args


def loading_data(args):
    print("Loading data")
    data = pd.read_csv(args.filepath) 
    return data



def pre_trained_model():
    print("Getting model")
    classifier = pipeline("text-classification", 
                      model="j-hartmann/emotion-english-distilroberta-base", 
                      return_all_scores=False) # Only return the emotion with the highest score 
    return classifier


def filtering(data):
    print("Filtering on real or fake")
    data_filtered_real = data[data['label'] == 'REAL']
    data_filtered_fake = data[data['label'] == 'FAKE']
    return data_filtered_real, data_filtered_fake


def data_cleaning(data, data_filtered_real, data_filtered_fake):
    print("Selecting column")
    all_texts = data["title"] # Only selecting column "title" and storing in headlines
    real_texts = data_filtered_real["title"]
    fake_texts = data_filtered_fake["title"]
    
    return all_texts, real_texts, fake_texts


def emotion_loops(classifier, all_texts, real_texts, fake_texts):
    print("Getting emotion label for: ")
    emotion_list = [] # Creating an empty list to store each emotion label.
    real_list = []
    fake_list = []
    print("All...")
    for line in all_texts: # For loop that goes through each headline in the list headlines, and uses the classifier.
        emotion_list.append(classifier(line)[0]["label"])  # append list #output[0]["label"]
        # using classifier to get emotion of each headline. The emotions are stored as dictionaries in a list. [0] gets the first dirctonary in the list ["label"] is they.
        # by only selecting label, I append it to the empty list emotion_list.
    print("Real...")   
    for line in real_texts: # The same logic in the code below
        real_list.append(classifier(line)[0]["label"])
    print("Fake...")
    for line in fake_texts: 
        fake_list.append(classifier(line)[0]["label"])
    
    return emotion_list, real_list, fake_list

def emotion_occurance(emotion_list, real_list, fake_list):
    print("Preparing for visualization")
    emotion_count = Counter(emotion_list) # Using counter from collections to get each occurance of the emotion 
    real_emotion_count = Counter(real_list) 
    fake_emotion_count = Counter(fake_list) 

    return emotion_count, real_emotion_count, fake_emotion_count # the count is stored as a dictionary

def splitting_dictionary(emotion_count, real_emotion_count, fake_emotion_count):
    print("Preparing for visualization")
    all_emotions = list(emotion_count.keys()) # Converting the keys to a list 
    all_count = list(emotion_count.values()) # Converting the values to a list 

    real_emotions = list(real_emotion_count.keys())
    real_count = list(real_emotion_count.values())

    fake_emotions = list(fake_emotion_count.keys())
    fake_count = list(fake_emotion_count.values())

    return all_emotions, all_count, real_emotions, real_count, fake_emotions, fake_count

    
def visualization_bar(emotions, count, title, folder = "figs"):
    print("Visualizing bar plots")
    plt.bar(range(len(emotions)), count, tick_label=emotions)
    plt.xlabel("Emotions")
    plt.ylabel("Count")
    plt.title(title)
    plt.show()

    filename = os.path.join(folder, title.replace(' ', '_') + '.png')
    plt.savefig(filename)
    plt.clf()

def visualization_scatter(all_emotions, all_count, real_emotions, real_count, fake_emotions, fake_count, folder = "figs"):
    print("Visualizing scatterplot")
    fig, ax = plt.subplots()

    ax.scatter(all_emotions, all_count, c='blue', label='All Emotions', marker= "o")
    #ax.plot(all_emotions, all_count, c='blue', label='All Emotions')

    ax.scatter(real_emotions, real_count, c='green', label='Real Emotions', marker= "+" )
    #ax.plot(real_emotions, real_count, c='green', label='Real Emotions')

    ax.scatter(fake_emotions, fake_count, c='red', label='Fake Emotions', marker= "_")
    #ax.plot(fake_emotions, fake_count, c='red', label='Fake Emotions')

    ax.set_xlabel('Emotion')
    ax.set_ylabel('Count')
    plt.legend()
    plt.title('Emotion Counts')

    filename = os.path.join(folder, "scatterplot" + '.png')
    plt.savefig(filename)
    plt.clf()

def main_function():
    args = input_parse()
    data = loading_data(args)
    classifier = pre_trained_model()
    data_filtered_real, data_filtered_fake = filtering(data)
    all_texts, real_texts, fake_texts = data_cleaning(data, data_filtered_real, data_filtered_fake)
    emotion_list, real_list, fake_list = emotion_loops(classifier, all_texts, real_texts, fake_texts)
    emotion_count, real_emotion_count, fake_emotion_count = emotion_occurance(emotion_list, real_list, fake_list)
    all_emotions, all_count, real_emotions, real_count, fake_emotions, fake_count = splitting_dictionary(emotion_count, real_emotion_count, fake_emotion_count)
    visualization_bar(all_emotions, all_count, "All Headings", folder = "figs")
    visualization_bar(real_emotions, real_count, "Real Headings", folder = "figs")
    visualization_bar(fake_emotions, fake_count, "Fake Headings", folder = "figs")
    visualization_scatter(all_emotions, all_count, real_emotions, real_count, fake_emotions, fake_count, folder = "figs")

if __name__ == "__main__":
    main_function()

















