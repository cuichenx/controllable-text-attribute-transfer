#Plot
import matplotlib.pyplot as plt
import seaborn as sns
from bleu import file_bleu
#Data Packages
import math
import pandas as pd
import numpy as np

#Progress bar
from tqdm import tqdm

#Counter
from collections import Counter

#Operation
import operator

#Natural Language Processing Packages
import re
import nltk

## Download Resources
nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tag import PerceptronTagger
from nltk.data import find

sns.set(rc={'figure.figsize':(5,3.5)})

# CHANGE FLIEPATH before running this locally

# Use vader to evaluated sentiment of reviews
def evalSentences(sentences, to_df=False, columns=[]):
    # Instantiate an instance to access SentimentIntensityAnalyzer class
    sid = SentimentIntensityAnalyzer()
    pdlist = []
    if to_df:
        for sentence in tqdm(sentences):
            ss = sid.polarity_scores(sentence)
            pdlist.append([sentence] + [ss['compound']])
        reviewDf = pd.DataFrame(pdlist)
        reviewDf.columns = columns
        return reviewDf
    
    else:
        for sentence in tqdm(sentences):
            print(sentence)
            ss = sid.polarity_scores(sentence)
            for k in sorted(ss):
                print('{0}: {1}, '.format(k, ss[k]), end='')
            print()


def getHistogram(df, measure, title, hue=None,  figsize=(5, 3)):
    if hue:
        sns_plot = sns.kdeplot(data=df, x=measure, hue=hue)
        # sns_plot = sns.histplot(data=df, x=measure, hue=hue)
    else:
        sns_plot = sns.histplot(data=df, x=measure)
    # sns_plot.set_title(title)
    sns_plot.set_xlabel("Value")
    sns_plot.set_ylabel("Density")
    plt.tight_layout()
    sns_plot.figure.savefig("{}.png".format(title))


def calculate_vader_ALE(filename=None):
    print("Evaluate ALE")
    if filename:
        file_path = "./{}.txt".format(filename)
    else:
        file_path  ="./outputext_step1_eps5.txt"
    file_path_neg  ="../../data/yelp/sentiment.test.0"
    file_path_pos  ="../../data/yelp/sentiment.test.1"
    
    review_file = open(file_path, "r")
    reviews = review_file.readlines()
    review_file.close()
    reviewDF = evalSentences(reviews, to_df=True, columns=['review','vader'])
    
    # sanity check
    assert(reviewDF.shape[0]==1000)
    
    neg_2_pos = (reviewDF[:500]['vader']>=0).sum()
    pos_2_neg = (reviewDF[500:]['vader']<=0).sum()
    
    acc = (neg_2_pos+pos_2_neg)/1000
    print("accuracy of changed sentences is {}".format(acc))
    print("accuracy of pos_to_neg sentences is {}".format(pos_2_neg/500))
    print("accuracy of neg_to_pos sentences is {}".format(neg_2_pos/500))
    
    review_file_neg = open(file_path_neg, "r")
    review_file_pos = open(file_path_pos, "r")
    reviews_neg = review_file_neg.readlines()
    reviews_pos = review_file_pos.readlines()
    review_file_neg.close()
    review_file_pos.close()
    
    reviewDF_neg = evalSentences(reviews_neg, to_df=True, columns=['review','vader'])
    reviewDF_pos = evalSentences(reviews_pos, to_df=True, columns=['review','vader'])
    
    # sanity check
    assert(reviewDF_neg.shape[0]==500)
    assert (reviewDF_pos.shape[0] == 500)
    
    pos_acc = (reviewDF_pos['vader']>=0).sum()
    neg_acc = (reviewDF_neg['vader']<=0).sum()
    
    org_acc = (pos_acc+neg_acc)/1000
    print("accuracy of original sentences is {}".format(org_acc))
    print("accuracy of original positive sentences is {}".format(pos_acc/500))
    print("accuracy of original negative sentences is {}".format(neg_acc/500))
    
    return reviewDF, reviewDF_pos, reviewDF_neg

def calculate_style_trans():
    print("Evaluate Style Transformer")
    file_path  ="./style_transformer.txt"
    
    review_file = open(file_path, "r")
    reviews_raw = review_file.readlines()
    review_file.close()
    
    reviews_pos_to_neg = [] # changed sentence
    reviews_neg_to_pos = []  # changed sentence
    reviews_pos = [] #original pos
    reviews_neg = [] #original neg
    pos_example = False
    neg_example = False
    for sent in reviews_raw:
        if sent.startswith("[raw  0.0]"):
            reviews_neg.append(sent[11:])
            neg_example = True
            pos_example = False
        elif sent.startswith("[raw  1.0]"):
            reviews_pos.append(sent[11:])
            neg_example = False
            pos_example = True
        elif sent.startswith("[rev  0.0]") and pos_example:
            reviews_pos_to_neg.append(sent[11:])
            pos_example = False
            neg_example = False
        elif sent.startswith("[rev  1.0]") and neg_example:
            reviews_neg_to_pos.append(sent[11:])
            pos_example = False
            neg_example = False
    assert (len(reviews_pos_to_neg) == 500)
    assert (len(reviews_neg_to_pos) == 500)
    assert (len(reviews_pos) == 500)
    assert (len(reviews_neg) == 500)

    reviewDF_pos_to_neg = evalSentences(reviews_pos_to_neg, to_df=True, columns=['review','vader'])
    reviewDF_neg_to_pos = evalSentences(reviews_neg_to_pos, to_df=True, columns=['review','vader'])

    neg_2_pos = (reviewDF_neg_to_pos['vader']>=0).sum()
    pos_2_neg = (reviewDF_pos_to_neg['vader']<=0).sum()
    
    acc = (neg_2_pos+pos_2_neg)/1000
    print("accuracy of changed sentences is {}".format(acc))
    print("accuracy of pos_to_neg sentences is {}".format(pos_2_neg/500))
    print("accuracy of neg_to_pos sentences is {}".format(neg_2_pos/500))
    
    reviewDF_neg = evalSentences(reviews_neg, to_df=True, columns=['review','vader'])
    reviewDF_pos = evalSentences(reviews_pos, to_df=True, columns=['review','vader'])
    
    # sanity check
    assert(reviewDF_neg.shape[0]==500)
    assert (reviewDF_pos.shape[0] == 500)
    
    pos_acc = (reviewDF_pos['vader']>=0).sum()
    neg_acc = (reviewDF_neg['vader']<=0).sum()
    
    org_acc = (pos_acc+neg_acc)/1000
    print("accuracy of original sentences is {}".format(org_acc))
    print("accuracy of original positive sentences is {}".format(pos_acc/500))
    print("accuracy of original negative sentences is {}".format(neg_acc/500))
    
    return reviewDF_pos_to_neg, reviewDF_neg_to_pos, reviewDF_pos, reviewDF_neg


def graph_ALE(reviewDF_ALE, reviewDF_pos_ALE, reviewDF_neg_ALE, color1, color2):
    reviewDF_pos_ALE['label'] = "POS"
    reviewDF_neg_ALE['label'] = "NEG"
    reviewDF_org = pd.concat((reviewDF_neg_ALE, reviewDF_pos_ALE), 0).reset_index(drop=True)
    assert (reviewDF_org['review']==reviewDF_ALE['review']).any() # there are definitely unchanged sentence, otherwise the ordering is wrong
    reviewDF_org = reviewDF_org.rename(columns={"vader":"vader_original"})
    reviewDF_ALE_all = pd.concat([reviewDF_ALE, reviewDF_org], axis=1, join="inner")
    reviewDF_ALE_all = reviewDF_ALE_all.loc[:,~reviewDF_ALE_all.columns.duplicated()]
    reviewDF_ALE_all['change in vader'] =  reviewDF_ALE_all['vader'] - reviewDF_ALE_all['vader_original']
    # getHistogram(reviewDF_ALE_all, 'change in vader', 'ALE change in vader score', hue="label")
    pal = dict(POS=color2, NEG=color1)
    sns_plot = sns.kdeplot(data=reviewDF_ALE_all, x='change in vader', hue="label", palette=pal)
    # sns_plot = sns.kdeplot(data=reviewDF_ALE_all, x='change in vader', color=color1)
    return sns_plot

def draw_transition_graph():
    palette = sns.color_palette("coolwarm", n_colors=10)
    i=0
    for filename in ["outputext_step1_eps0.5", "outputext_step1_eps2", "outputext_step1_eps3", "outputext_step1_eps4", "outputext_step1_eps5"]:
        color1 = palette[4-i]
        color2 = palette[i+5]
        i+=1
        reviewDF_ALE, reviewDF_pos_ALE, reviewDF_neg_ALE = calculate_vader_ALE(filename)
        plot = graph_ALE(reviewDF_ALE, reviewDF_pos_ALE, reviewDF_neg_ALE, color1, color2)
    plot.figure.savefig("ALE transition")

def graph_ALE_vader():
    reviewDF_ALE, reviewDF_pos_ALE, reviewDF_neg_ALE = calculate_vader_ALE()
    reviewDF_pos_ALE['label'] = "POS → NEG"
    reviewDF_neg_ALE['label'] = "NEG → POS"
    reviewDF_org = pd.concat((reviewDF_neg_ALE, reviewDF_pos_ALE), 0).reset_index(drop=True)
    assert (reviewDF_org['review']==reviewDF_ALE['review']).any() # there are definitely unchanged sentence, otherwise the ordering is wrong
    reviewDF_org = reviewDF_org.rename(columns={"vader":"vader_original"})
    reviewDF_ALE_all = pd.concat([reviewDF_ALE, reviewDF_org], axis=1, join="inner")
    reviewDF_ALE_all = reviewDF_ALE_all.loc[:,~reviewDF_ALE_all.columns.duplicated()]
    reviewDF_ALE_all['change in vader'] =  reviewDF_ALE_all['vader'] - reviewDF_ALE_all['vader_original']
    getHistogram(reviewDF_ALE_all, 'change in vader', 'change in vader score (ALE)', hue="label")

def graph_ST_vader():
    reviewDF_pos_to_neg, reviewDF_neg_to_pos, reviewDF_pos, reviewDF_neg = calculate_style_trans()
    reviewDF_pos['label'] = "POS → NEG"
    reviewDF_neg['label'] = "NEG → POS"
    reviewDF_org = pd.concat((reviewDF_neg, reviewDF_pos), 0).reset_index(drop=True)
    reviewDF_trans = pd.concat((reviewDF_neg_to_pos, reviewDF_pos_to_neg), 0).reset_index(drop=True)
    assert (reviewDF_org['review']==reviewDF_trans['review']).any() # there are definitely unchanged sentence, otherwise the ordering is wrong
    reviewDF_org = reviewDF_org.rename(columns={"vader":"vader_original"})
    reviewDF_trans_all = pd.concat([reviewDF_trans, reviewDF_org], axis=1, join="inner")
    reviewDF_trans_all = reviewDF_trans_all.loc[:,~reviewDF_trans_all.columns.duplicated()]
    reviewDF_trans_all['change in vader'] =  reviewDF_trans_all['vader'] - reviewDF_trans_all['vader_original']
    getHistogram(reviewDF_trans_all, 'change in vader', 'change in vader score (StyleTransformer)', hue="label")

def calculate_vader_ALE_human():
    print("Evaluate ALE Ground Truth")
    file_path = "D:\year4\CSC413\project\controllable-text-attribute-transfer\data\yelp\human.txt"
    
    review_file = open(file_path, "r")
    reviews = review_file.readlines()
    review_file.close()
    reviewDF = evalSentences(reviews, to_df=True, columns=['review', 'vader'])
    
    # sanity check
    assert (reviewDF.shape[0] == 1000)
    
    neg_2_pos = (reviewDF[:500]['vader'] >= 0).sum()
    pos_2_neg = (reviewDF[500:]['vader'] <= 0).sum()
    
    acc = (neg_2_pos + pos_2_neg) / 1000
    print("accuracy of changed sentences is {}".format(acc))
    print("accuracy of pos_to_neg sentences is {}".format(pos_2_neg / 500))
    print("accuracy of neg_to_pos sentences is {}".format(neg_2_pos / 500))
    
    
    return reviewDF

graph_ALE_vader()
graph_ST_vader()
draw_transition_graph()