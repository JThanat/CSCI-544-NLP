import json
import os
import re
import sys
import numpy as np
import random

path_to_train = sys.argv[1]

stop_words = {}
# stop_words = {"the", "and", "a", "to", "in", "was", "i", "of", "for", "room", "at", "it", "this", "my", "with", "is", "that", "were", "on", "had", "we", "have", "be", "from", "when", "all", "you", "our", "as", "so", "stayed", "hotels"}
neg_dict = {
        "can't": "can not",
        "couldn't": "could not",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "won't": "will not",
        "would't": "would not",
        "shouldn't": "should not",
        "isn't": "is not",
        "aren't": "are not",
    }

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
                        doc += line.lower()
                    sub_doc = multiple_replace(neg_dict, doc)
                    docs.append(sub_doc)
    return docs


def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


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

    count_token = dict(filter(lambda elem: elem[1] >= 5, count_token.items()))
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
    for i in range(0, max_iter):
        random.shuffle(feature_vectors)
        converged = True
        for y, xd in feature_vectors:
            ac = np.dot(xd, wd)
            if ac * y <= 0:
                converged = False
                wd = wd + y*xd
                b = b + y
        if converged:
            # print("converge at round {}".format(i))
            break
    return wd, b


def train_average_classifier(feat_len, max_iter, feature_vectors):
    wd = np.zeros(feat_len)
    b = 0
    cached_w = np.zeros(feat_len)
    cached_b = 0
    c = 1
    for i in range(0, max_iter):
        random.shuffle(feature_vectors)
        converged = True
        for y, xd in feature_vectors:
            ac = np.dot(xd, wd)
            if ac * y <= 0:
                converged = False
                wd = wd + y*xd
                b = b + y
                cached_w = cached_w + y*c*xd
                cached_b = cached_b + y*c
            c += 1
        if converged:
            print("converge at round {}".format(i))
            break
    return wd - cached_w/c, b - cached_b/c


neg_dec = os.scandir(os.path.join(path_to_train, "negative_polarity/deceptive_from_MTurk"))
neg_tr = os.scandir(os.path.join(path_to_train, "negative_polarity/truthful_from_Web"))
pos_dec = os.scandir(os.path.join(path_to_train, "positive_polarity/deceptive_from_MTurk"))
pos_tr = os.scandir(os.path.join(path_to_train, "positive_polarity/truthful_from_TripAdvisor"))

neg_dec_docs = read_file(neg_dec)
neg_tr_docs = read_file(neg_tr)
pos_dec_docs = read_file(pos_dec)
pos_tr_docs = read_file(pos_tr)

#### Vanilla Classifier ####
max_iter = 100
random.seed(99)
# Feature Selection
features_candidate = feature_selection(neg_dec_docs + neg_tr_docs + pos_dec_docs + pos_tr_docs)[:1500]
features_candidate = [w for w, c in features_candidate]
fn = set([k for k, _ in feature_selection(neg_dec_docs + neg_tr_docs)])
fp = set([k for k, _ in feature_selection(pos_dec_docs + pos_tr_docs)])
fd = set([k for k, _ in feature_selection(neg_dec_docs + pos_dec_docs)])
ft = set([k for k, _ in feature_selection(neg_tr_docs + pos_tr_docs)])
fpn = fn.union(fp)
ftd = fd.union(ft)
fc_pn_list = sorted(list(fpn))
fc_td_list = sorted(list(ftd))
# Truthful Deceptive
training_vector_td = vector_construction(1, neg_tr_docs + pos_tr_docs, fc_td_list)
training_vector_td = training_vector_td + vector_construction(-1, neg_dec_docs + pos_dec_docs, fc_td_list)
w_td, b_td = train_classifier(len(fc_td_list), max_iter, training_vector_td)
# Positive Negative
training_vector_pn = vector_construction(1, pos_dec_docs + pos_tr_docs, fc_pn_list)
training_vector_pn = training_vector_pn + vector_construction(-1, neg_dec_docs + neg_tr_docs, fc_pn_list)
w_pn, b_pn = train_classifier(len(fc_pn_list), max_iter, training_vector_pn)

#### Average Classifier ####
# Truthful Deceptive
w_td_avg, b_td_avg = train_classifier(len(fc_td_list), max_iter, training_vector_td)
# Positive Negative
w_pn_avg, b_pn_avg = train_classifier(len(fc_pn_list), max_iter, training_vector_pn)


average_classifier_model = {"w": {}, "b": {}, "features": {}}
average_classifier_model["b"]["truthful_deceptive"] = b_td_avg
average_classifier_model["b"]["positive_negative"] = b_pn_avg
average_classifier_model["w"]["positive_negative"] = w_pn_avg.tolist()
average_classifier_model["w"]["truthful_deceptive"] = w_td_avg.tolist()
average_classifier_model["features"]["truthful_deceptive"] = fc_td_list
average_classifier_model["features"]["positive_negative"] = fc_pn_list
with open("averagedmodel.txt", "w", encoding="utf-8") as txt_writer:
    json.dump(average_classifier_model, txt_writer)

vanilla_classifier_model = {"w": {}, "b": {}, "features": {}}
vanilla_classifier_model["b"]["truthful_deceptive"] = b_td
vanilla_classifier_model["b"]["positive_negative"] = b_pn
vanilla_classifier_model["w"]["positive_negative"] = w_pn.tolist()
vanilla_classifier_model["w"]["truthful_deceptive"] = w_td.tolist()
vanilla_classifier_model["features"]["truthful_deceptive"] = fc_td_list
vanilla_classifier_model["features"]["positive_negative"] = fc_pn_list
with open("vanillamodel.txt", "w", encoding="utf-8") as txt_writer:
    json.dump(vanilla_classifier_model, txt_writer)