# Notes: Any posts/comments are removed that are removed or deleted. If the entire post is deleted all comments are too, but if the comments are deleted the rest of the post remains intact.
# All comments from layman can be toggled to be removed or not.
# RC are comments, RS are posts

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


# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
# model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

from transformers import AutoTokenizer, RobertaModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_punkt():
    import nltk
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('punkt')

def gzip_to_json(input,output): #input is askdocs_input/, output is askdocs_output/
    import gzip
    import shutil
    for subdir, dirs, files in os.walk(input):
        for file in files:
            print(file)
            with gzip.open(input+ "/"+file, 'rb') as file_in:
                with open(output + "/"+file, 'wb') as file_out:
                    shutil.copyfileobj(file_in, file_out)

# # Pair up comments with posts, return resultant dict
def gather_pairs(input_file):
    with open("askdocs_output/"+ input_file) as f:
        json_post_data = [json.loads(line) for line in f]

    post_data = {}
    for item in json_post_data:
        post_data.update({item["id"]: {"post_id": item["id"],"post_title":item["title"], "post_text": item["selftext"],"post_time_written": item["author_created_utc"],"post_upvotes": item["score"],"post_upvote_ratio": item["upvote_ratio"], "post_award_count": item["total_awards_received"], "post_link": item["permalink"]}})

    with open("askdocs_output/" + "RC"+input_file[2:]) as f:
        # print("RC"+input_file[2:])
        json_comment_data = [json.loads(line) for line in f]

    comment_data  = {}
    for item in json_comment_data:
        if item["author"] != "AutoModerator":
            previous_list = comment_data.get(item["link_id"][3:],"no prior entries")
            if previous_list == "no prior entries": 
                comment_data.update({item["link_id"][3:]: [{"parent_id": item["link_id"][3:],"comment_id": item["id"],"comment_text": item["body"], "comment_flair": item["author_flair_text"],"comment_controversiality": item["controversiality"],"comment_award_count": item["total_awards_received"],"comment_upvotes": item["score"],"comment_time_written": item["author_created_utc"], "comment_link": item["permalink"]}]})
            else:
                previous_list.append({"parent_id": item["link_id"][3:],"comment_id": item["id"],"comment_text": item["body"], "comment_flair": item["author_flair_text"],"comment_controversiality": item["controversiality"],"comment_award_count": item["total_awards_received"],"comment_upvotes": item["score"],"comment_time_written": item["author_created_utc"], "comment_link": item["permalink"]})
                comment_data.update({item["link_id"][3:]: previous_list})

    post_comment_pairs = {}

    for id in list(post_data.keys()):
        if id in comment_data:
            comment_list = comment_data[id]
            if len(comment_list) > 1:
                new_comment_list = []
                for comment in comment_list:
                    if post_data[id]["post_text"] not in {"[deleted]","[removed]"}:
                        # if comment["comment_flair"] not in {"Layperson/not verified as healthcare professional.","Layperson/not verified as healthcare professional","This user has not yet been verified.","None"}:
                            if comment["comment_text"] not in {"[deleted]","[removed]"}:
                                new_comment_list.append(comment)            
                if len(new_comment_list) > 0:
                    post_comment_pairs.update({id:{"post": post_data[id], "list_of_comments": new_comment_list}})

            else:
                if not post_data[id]["post_text"] in {"[deleted]","[removed]"}:
                    #if not comment_list[0]["comment_flair"] in {"Layperson/not verified as healthcare professional.","Layperson/not verified as healthcare professional","This user has not yet been verified.","None"}:
                        if not comment_list[0]["comment_text"] in {"[deleted]","[removed]"}:
                            post_comment_pairs.update({id:{"post": post_data[id], "list_of_comments": [comment_list[0]]}})
        
    # for i in list(post_comment_pairs.items())[0:5]:
    #     print("\n----------")
    #     print('Post title: "' + i[1]['post']["post_title"]+'"')
    #     print("Post upvotes: " + str(i[1]['post']["post_upvotes"]))
    #     print('Post text: "' + i[1]['post']["post_text"]+'"')
    #     print("\n")
        
    #     count = 1
    #     for j in i[1]['list_of_comments']:
    #         print("__Comment #" + str(count))
    #         print("Upvotes: "+ str(j["comment_upvotes"]))
    #         print('Comment text: "'+ j["comment_text"]+'"\n')
    #         count +=1

    return post_comment_pairs

# From Ungziped files gather pairs of comments and posts, write to folder, input is askdocs_output, output is #askdocs_pairs
def gather_pairs_iterator(input_unzipped, output_pairs):
    for subdir, dirs, files in os.walk(input_unzipped):
        for file in files:
            if file[0:2] == "RS":
                print(file)
                with open(output_pairs+"/"+file, "w") as fp:
                    post_comment_pairs = gather_pairs(file)
                    json.dump(post_comment_pairs , fp) 

# # From Comment pairs, lookup a certain phrase, return the count and write to a file. 
def lookup_iterator(search_pairs_folder,search_phrase):
    for subdir, dirs, files in os.walk(search_pairs_folder):
        keyword_count = 0
        for file in files:
            print(file)
            with open(search_pairs_folder+"/"+file) as json_file:
                data = json.load(json_file)
                keyword_count = lookup(data,search_phrase)
            print(keyword_count)
            #30558

# # From Custom Dict object, search for a certain keyword and return the count, write to a seperate file.
def lookup(post_comment_pairs, phrase): 
    keyword_count = 0
    with open("askdocs_keyword_output/"+file[0:27], "w") as fp:  
        matching_data = []  # List to store matching data
        for i in list(post_comment_pairs.items()):
            if phrase in i[1]['post']["post_text"]:
                keyword_count += 1
                matching_data.append(i)

            
            for j in i[1]['list_of_comments']:
                if phrase in j["comment_text"]:
                    keyword_count += 1
                    matching_data.append(i)
        json.dump(matching_data, fp)
    return keyword_count

def sentiment_analysis(keyword_search_folder, sentiment_model):
    global total_pairs
    total_pairs = 0
    global tokenizer
    global model

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


    count = 0
    for subdir, dirs, files in os.walk(keyword_search_folder):
        for file in files:
            print(file)


            with open(keyword_search_folder+"/"+file, "r") as data_file:
                data = json.load(data_file)
                
                for record in data:
                    if total_pairs > 13681 or count > 13681:
                        return "exited analysis loop"
                    
                    count+= 1
                    post = record[1].get('post', {})
                    comments = record[1].get('list_of_comments', [])

                    #Entire post sentiment analysis
                    # print(post["post_text"])
                    post_sentences = sent_tokenize(post["post_text"])

                    #individual line sentiment
                    # print("______Post______")
                    for sentence in post_sentences:
                        # print("_________")
                        # print(sentence)
                        return_data = sentiment_function(sentence)

                    #individual comment sentiment
                    # print("\n____Comments_____")
                    for comment in comments:
                        # print("____next comment_____\n")
                        comment_sentences = sent_tokenize(comment["comment_text"])
                        for comment_sentence in comment_sentences:
                            # print("________")
                            # print(comment_sentence)
                            return_data = sentiment_function(comment_sentence)

                    # print("_________________Next Post________________________\n\n\n")
                    print("Run time so far: %s seconds ---" % (time.time() - start_time))
                    print(f"count: {total_pairs + count} of 53932")
                total_pairs += count

#Running sentiment analysis using Stanford CoreNLP Python port Stanza
def stanza_sentiment(input_sentence):
    sentiment_dict = {0:"Negative", 1: "Neutral", 2: "Positive"}
    doc = nlp(input_sentence)
    for sentence_iterator in doc.sentences:
    #     print(f"Sentence sentiment -> {sentiment_dict.get(sentence_iterator.sentiment)}")
        return sentiment_dict.get(sentence_iterator.sentiment)

def vader_sentiment(input_sentence):
    score = analyzer.polarity_scores(input_sentence)
    # print(score)
    return score

#Running sentiment analysis using Python nlp library Pattern
def sentimentr_sentiment(input_sentence):
    score = s.get_polarity_score(input_sentence, subjectivity=True)
    # print(score)
    return score

def roberta_sentiment(input_sentence):
    LABELS = {0: 'negative', 1: 'neutral', 2: 'positive'}

    inputs = tokenizer(input_sentence, return_tensors="pt")
    outputs = model(**inputs)["logits"][0].detach().tolist() #list of logits [.3,.5,.-9]

    softmax_values = np.exp(outputs) / np.sum(np.exp(outputs))
    label = LABELS[softmax_values.argmax()]
    print(label)
    return outputs,softmax_values,label

def main():
    global start_time
    start_time = time.time()

    # download_punkt()

    # gzip_to_json("askdocs_input", "askdocs_output")

    # gather_pairs_iterator("askdocs_output", "askdocs_pairs")
    
    # lookup_iterator("askdocs_pairs", "pain")

    #Still need an output way to store sentiment data.
    sentiment_analysis("askdocs_keyword_output", "Generic")

    print("--- %s seconds ---" % (time.time() - start_time))


main()