import json
import math
import sys
import os
import re

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
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now","room", "hotel", "chicago", "stay", "rooms", "staff", "stayed", "Hotel", "michigan"}


def tokenize(doc):
    # filter stopwords later
    token_list = [x.strip() for x in re.sub(r"[^a-zA-Z0-9 ]", "", doc).split(" ") if len(x) > 0]
    tokens = set(token_list)
    for token in token_list:
        if token.lower() in stop_words and token in tokens:
            tokens.remove(token)
    return tokens


def concat_text_in_class(class_docs):
    text = ""
    for doc in class_docs:
        text += doc.strip()
    return [x.strip() for x in re.sub(r"[^a-zA-Z0-9 ]", "", text).split(" ") if len(x) > 0]


def create_freq_dict(concat_text):
    freq_d = {}
    for t in concat_text:
        if t not in freq_d:
            freq_d[t] = 0
        freq_d[t] += 1
    return freq_d


def train_classifier(docs, condprob, features):
    nb_class = docs.keys()
    prior = {}
    vocab = set([])
    for _class in docs:
        for doc in docs[_class]:
            vocab.update(tokenize(doc))
    # vocab = vocab.intersection(features)
    n = sum(len(d) for c, d in docs.items())
    for c in nb_class:
        nc = len(docs[c])
        prior[c] = nc / n
        # vocab = tokenize(docs[c])
        concat_text = concat_text_in_class(docs[c])
        freq_dict = create_freq_dict(concat_text)
        for term in vocab:
            tct = freq_dict[term] if term in freq_dict else 0
            if term not in condprob:
                condprob[term] = {}
            condprob[term][c] = (tct + 1) / (len(concat_text) + len(vocab))
        condprob["#"][c] = 1 / (len(concat_text) + len(vocab))
    return vocab, prior


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


def compute_feature_utility(docs, t, nb_class):
    # Ntc --> t: document contain t, c: doc in class c
    # Ex: N10 -> doc that contains t (et = 1) and doc is not in class c (ec = 0)
    N00 = 0
    N11 = 0
    N01 = 0
    N10 = 0
    for c in nb_class:
        for doc in docs[c]:
            if c == "truthful" or c == "positive":
                if t in doc:
                    N11 += 1
                else:
                    N01 += 1
            else:
                if t in doc:
                    N10 += 1
                else:
                    N00 += 1
    if N00 == 0:
        N00 += 1
    if N01 == 0:
        N01 += 1
    if N10 == 0:
        N10 += 1
    if N11 == 0:
        N11 += 1
    N = N00 + N01 + N10 + N11
    U = N11 * math.log2((N * N11) / ((N10 + N11) * (N01 + N10))) / N
    U += N01 * math.log2((N * N01) / ((N00 + N01) * (N10 + N11))) / N
    U += N10 * math.log2((N * N10) / ((N10 + N11) * (N00 + N10))) / N
    U += N00 * math.log2((N * N00) / ((N00 + N01) * (N00 + N01))) / N
    return U


def select_feature(docs, k):
    nb_class = docs.keys()
    vocab = set([])
    L = []
    for _class in docs:
        for doc in docs[_class]:
            vocab.update(doc)
    for t in vocab:
        U = compute_feature_utility(docs, t, nb_class)
        L.append((U, t))
    L.sort(reverse=True)
    print(L)
    return L[:k]

neg_dec = os.scandir(os.path.join(path_to_train, "negative_polarity/deceptive_from_MTurk"))
neg_tr = os.scandir(os.path.join(path_to_train, "negative_polarity/truthful_from_Web"))
pos_dec = os.scandir(os.path.join(path_to_train, "positive_polarity/deceptive_from_MTurk"))
pos_tr = os.scandir(os.path.join(path_to_train, "positive_polarity/truthful_from_TripAdvisor"))

neg_dec_docs = read_file(neg_dec)
neg_tr_docs = read_file(neg_tr)
pos_dec_docs = read_file(pos_dec)
pos_tr_docs = read_file(pos_tr)

neg_dec_tokenized_docs = []
for doc in neg_dec_docs:
    neg_dec_tokenized_docs.append(tokenize(doc))

neg_tr_tokenized_docs = []
for doc in neg_tr_docs:
    neg_tr_tokenized_docs.append(tokenize(doc))

pos_dec_tokenized_docs = []
for doc in pos_dec_docs:
    pos_dec_tokenized_docs.append(tokenize(doc))

pos_tr_tokenized_docs = []
for doc in pos_tr_docs:
    pos_tr_tokenized_docs.append(tokenize(doc))

# k = 100000
k = int(sys.argv[2])
features = select_feature({"truthful": neg_tr_tokenized_docs + pos_tr_tokenized_docs, "deceptive": neg_dec_tokenized_docs + pos_dec_tokenized_docs}, k)
features_set_truthful = set([v for _, v in features])

features = select_feature({"positive": pos_dec_tokenized_docs + pos_tr_tokenized_docs, "negative": neg_dec_tokenized_docs + neg_tr_tokenized_docs}, k)
features_set_positive = set([v for _, v in features])

condprob = {"#": {}}
vocab, prior_td = train_classifier({"truthful": neg_tr_docs + pos_tr_docs, "deceptive": neg_dec_docs + pos_dec_docs},
                                   condprob, features_set_truthful)
_, prior_pn = train_classifier({"positive": pos_dec_docs + pos_tr_docs, "negative": neg_dec_docs + neg_tr_docs},
                               condprob, features_set_positive)

classifier_params = ["vocab", "prior", "condprob"]
classifier_model = dict.fromkeys(classifier_params)

classifier_model["vocab"] = list(vocab)
classifier_model["prior"] = {**prior_pn, **prior_td}
classifier_model["condprob"] = condprob
classifier_model["truthful"] = {"features": list(features_set_truthful)}
classifier_model["positive"] = {"features": list(features_set_positive)}

with open("nbmodel.txt", "w", encoding="utf-8") as txt_writer:
    json.dump(classifier_model, txt_writer)

# neg_dec.close()
# neg_tr.close()
# pos_dec.close()
# pos_tr.close()
