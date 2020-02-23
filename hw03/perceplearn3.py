import json
import os
import re
import sys
import numpy as np
import random

path_to_train = sys.argv[1]


stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now","room", "hotel", "chicago", "stay", "rooms", "staff", "stayed", "Hotel", "michigan", "would"}


def get_all_file_path(path, depth):
    if os.path.isfile(path) and ".txt" in path and "README" not in path and depth == 4:
        return [path]
    if os.path.isdir(path):
        result = []
        for entry in os.scandir(path):
            result.extend(get_all_file_path(entry.path, depth + 1))
        return result
    return []

def read_file(class_folder):
    docs = []
    for fold in class_folder:
        if fold.is_dir():
            folder = os.scandir(fold.path)
            for training_file in folder:
                if ".txt" not in training_file.path or "README" in training_file.path:
                    continue
                with open(training_file.path, "r", encoding="utf-8") as file_reader:
                    doc = ""
                    for line in file_reader:
                        doc += line
                    docs.append(doc)
    return docs


def tokenize(doc):
    # filter stopwords later
    token_list = [x.strip() for x in re.sub(r"[^a-zA-Z ]", "", doc).split(" ") if len(x) > 0]
    tokens = set(token_list)
    for token in token_list:
        if token.lower() in stop_words and token in tokens:
            tokens.remove(token)
    return tokens

def feature_selection(docs):
    count_token = {}
    for i, doc in enumerate(docs):
        token_set = tokenize(doc)
        for t in token_set:
            if t not in count_token:
                count_token[t] = 0
            count_token[t] += 1

    sorted_x = sorted(count_token.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_x

def vector_construction(_class_value, docs, features):
    vect_list = []
    for doc in docs:
        word_list = [x.strip() for x in re.sub(r"[^a-zA-Z ]", "", doc).split(" ") if len(x) > 0]
        count_term_freq = dict.fromkeys(word_list, 0)
        for w in word_list:
            count_term_freq[w] += 1

        feature_vect = np.zeros(len(features))
        for i, f in enumerate(features):
            feature_vect[i] = count_term_freq.get(f, 0.0)
        vect_list.append((_class_value, feature_vect))
    return vect_list


def train_classifier(feat_len, max_iter, feature_vectors):
    wd = np.zeros(feat_len)
    b = 0
    random.seed(99)
    for i in range(0, max_iter):
        random.shuffle(feature_vectors)
        updated = False
        for y, xd in feature_vectors:
            ac = np.dot(xd, wd)
            if ac * y <= 0:
                updated = True
                wd = wd + y*xd
                b = b + y
        if not updated:
            break
    return wd, b


neg_dec = os.scandir(os.path.join(path_to_train, "negative_polarity/deceptive_from_MTurk"))
neg_tr = os.scandir(os.path.join(path_to_train, "negative_polarity/truthful_from_Web"))
pos_dec = os.scandir(os.path.join(path_to_train, "positive_polarity/deceptive_from_MTurk"))
pos_tr = os.scandir(os.path.join(path_to_train, "positive_polarity/truthful_from_TripAdvisor"))

neg_dec_docs = read_file(neg_dec)
neg_tr_docs = read_file(neg_tr)
pos_dec_docs = read_file(pos_dec)
pos_tr_docs = read_file(pos_tr)

#### Vanilla Classifier ####
max_iter = 2000
# Feature Selection
features_candidate = feature_selection(neg_dec_docs + neg_tr_docs + pos_dec_docs + pos_tr_docs)[:1500]
features_candidate = [w for w, c in features_candidate]
# Truthful Deceptive
feature_vector = vector_construction(1, neg_tr_docs + pos_tr_docs, features_candidate)
feature_vector = feature_vector + vector_construction(-1, neg_dec_docs + pos_dec_docs, features_candidate)
w_td, b_td = train_classifier(len(features_candidate), max_iter, feature_vector)
# Positive Negative
feature_vector = vector_construction(1, pos_dec_docs + pos_tr_docs, features_candidate)
feature_vector = feature_vector + vector_construction(-1, neg_dec_docs + neg_tr_docs, features_candidate)
w_pn, b_pn = train_classifier(len(features_candidate), max_iter, feature_vector)

#### Average Classifier ####
# Truthful Deceptive
# train_classifier()
# Positive Negative
# train_classifier()


# average_classifier_params = ["", "", ""]
# average_classifier_model = dict.fromkeys(average_classifier_params)
# with open("averagedmodel.txt", "w", encoding="utf-8") as txt_writer:
#     json.dump(average_classifier_model, txt_writer)
vanilla_classifier_model = {"w":{}, "b":{}}
vanilla_classifier_model["b"]["truthful_deceptive"] = b_td
vanilla_classifier_model["b"]["positive_negative"] = b_pn
vanilla_classifier_model["w"]["positive_negative"] = w_pn.tolist()
vanilla_classifier_model["w"]["truthful_deceptive"] = w_td.tolist()
vanilla_classifier_model["features"] = features_candidate
with open("vanillamodel.txt", "w", encoding="utf-8") as txt_writer:
    json.dump(vanilla_classifier_model, txt_writer)