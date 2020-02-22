import math
import os
import sys
import re
import json

path_to_test = sys.argv[1]

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


def classify(doc, model, nbclass):
    tokens = tokenize(doc)
    w = tokens.intersection(set(model["vocab"]))
    # w = tokens
    score = {}
    for c in nbclass:
        if c == "truthful" or c =="deceptive":
            w = tokens.intersection(set(model["truthful"]["features"]))
        else:
            w = tokens.intersection(set(model["positive"]["features"]))
        score[c] = math.log(model["prior"][c])
        for t in w:
            if t in model["condprob"] and c in model["condprob"][t]:
                score[c] += math.log(model["condprob"][t][c])
            else:
                score[c] += math.log(model["condprob"]["#"][c])

    td = "deceptive" if score["deceptive"] > score["truthful"] else "truthful"
    pn = "negative" if score["negative"] > score["positive"] else "positive"
    return td, pn


def get_all_file_path(path, depth):
    if os.path.isfile(path) and ".txt" in path and "README" not in path and depth == 4:
        return [path]
    if os.path.isdir(path):
        result = []
        for entry in os.scandir(path):
            result.extend(get_all_file_path(entry.path, depth + 1))
        return result
    return []


with open("nbmodel.txt", "r", encoding="utf-8") as txt_reader:
    model = json.load(txt_reader)

# test_dir = os.scandir(path_to_test)
test_file_path = get_all_file_path(path_to_test, 0)

output = ""
for test_file in test_file_path:
    with open(test_file, "r", encoding="utf-8") as file_reader:
        doc = ""
        for line in file_reader:
            doc += line
        td, pn = classify(doc, model, ["truthful", "deceptive", "positive", "negative"])
        output += td + " " + pn + " " + test_file + "\n"

with open("nboutput.txt", "w", encoding="utf-8") as txt_writer:
    txt_writer.write(output.strip())

# test_dir.close()
