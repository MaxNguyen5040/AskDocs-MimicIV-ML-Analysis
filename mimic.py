import csv  
import time
#total rows: 98972025

import json
import time
import os
import stanza
import nltk
from nltk.tokenize import sent_tokenize
from sentimentr.sentimentr import Sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import numpy as np

from transformers import AutoTokenizer, RobertaModel, AutoModelForSequenceClassification
import torch


def sentiment_setup(sentiment_model):
    global total_pairs
    global tokenizer
    global model
    global sentiment_function

    models = {"Stanza": stanza_sentiment, "Sentimentr": sentimentr_sentiment, "Vader": vader_sentiment, "Roberta": roberta_sentiment, "DistilRoberta": roberta_sentiment, "Generic": roberta_sentiment}
    sentiment_function = models.get(sentiment_model)

    if sentiment_model == "Stanza":
        global nlp
        nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment', tokenize_no_ssplit=True)

    if sentiment_model == "Sentimentr":
        global s
        s = Sentiment

    if sentiment_model == "Vader":
        global analyzer
        analyzer = SentimentIntensityAnalyzer()

    if sentiment_model == "Roberta":
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

    if sentiment_model == "DistilRoberta":
        tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
        model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

    if sentiment_model == "Generic":
        tokenizer = AutoTokenizer.from_pretrained("Seethal/sentiment_analysis_generic_dataset")
        model = AutoModelForSequenceClassification.from_pretrained("Seethal/sentiment_analysis_generic_dataset")
        
 
def keyword_iterator():
    with open("discharge.csv") as csvfile:  
        data = csv.DictReader(csvfile)
        keyword_count = 0
        for row in data:
            if keyword_count > 30558:
                break
            
            if "pain" in row["text"]:
                keyword_count += 1

                new_dict = {}
                for i in row:
                    new_dict[i] = row[i]

                print("___________________________________")

                with open('mimic_keyword.jsonl', 'a') as jsonl_file:
                    jsonl_file.write(json.dumps(new_dict) + '\n')

            print("\n\n\nRun time so far: %s seconds" % (time.time() - start_time))
            print(" _______  Amount of notes with keyword: "+ str(keyword_count)+"_____________")


def sentiment_analysis_iterator():
    with open('mimic_keyword.jsonl', 'r') as jsonl_file:
        count = 0
        for line in jsonl_file:
            count+= 1
            json_object = json.loads(line)

            note_sentences = sent_tokenize(json_object["text"])
            for sentence in note_sentences:
                print("\n")
                print('Sentence: "'+sentence+'"')
                if len(sentence) > 500:
                    print("Sentence over 500 chars")
                    split_sentence = sentence.split()
                    sentence_length = len(split_sentence)
                    sentence1 = " ".join(split_sentence[0:int(sentence_length/2)])
                    sentence2 = " ".join(split_sentence[int(sentence_length/2):])

                    return_data = sentiment_function(sentence1)
                    return_data = sentiment_function(sentence2)
                else:
                    return_data = sentiment_function(sentence)
                #eventually should convert these into a list of sentences and do like len/500 for amount

                print("\n\n\nRun time so far: %s seconds" % (time.time() - start_time))
            print(" _______  Count: "+ str(count)+"_____________")

#Running sentiment analysis using Stanford CoreNLP Python port Stanza
def stanza_sentiment(input_sentence):
    sentiment_dict = {0:"Negative", 1: "Neutral", 2: "Positive"}
    doc = nlp(input_sentence)
    for sentence_iterator in doc.sentences:
        # print(f"Sentence sentiment -> {sentiment_dict.get(sentence_iterator.sentiment)}")
        return sentiment_dict.get(sentence_iterator.sentiment)

def vader_sentiment(input_sentence):
    score = analyzer.polarity_scores(input_sentence)
    print(score)
    return score

#Running sentiment analysis using Python nlp library Pattern
def sentimentr_sentiment(input_sentence):
    score = s.get_polarity_score(input_sentence, subjectivity=True)
    return score

def roberta_sentiment(input_sentence):
    LABELS = {0: 'negative', 1: 'neutral', 2: 'positive'}

    inputs = tokenizer(input_sentence, return_tensors="pt")
    outputs = model(**inputs)["logits"][0].detach().tolist() #list of logits [.3,.5,.-9]

    softmax_values = np.exp(outputs) / np.sum(np.exp(outputs))
    label = LABELS[softmax_values.argmax()]
    # print(label)
    return outputs,softmax_values,label

def main():
    global start_time
    start_time = time.time()

    # mimic_iterator()

    sentiment_setup("Generic")

    sentiment_analysis_iterator()

    print("--- %s seconds ---" % (time.time() - start_time))






main()