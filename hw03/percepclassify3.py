import json
import os
import re
import sys

import numpy as np

path_to_model = sys.argv[1]
path_to_test = sys.argv[2]

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


def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


def vector_construction(doc, features):
    word_list = [x.strip() for x in re.sub(r"[^a-zA-Z ]", "", doc).split(" ") if len(x) > 0]
    count_term_freq = dict.fromkeys(word_list, 0)
    for w in word_list:
        count_term_freq[w] += 1

    feature_vect = np.zeros(len(features))
    for i, f in enumerate(features):
        feature_vect[i] = count_term_freq.get(f, 0.0)
    return feature_vect


def classify(doc, model, pclass):
    score = {}
    for c in pclass:
        v = vector_construction(doc, model["features"][c])
        predicted = np.dot(model["w"][c], v) + model["b"][c]
        score[c] = 1 if predicted > 0 else 0
    td = "truthful" if score["truthful_deceptive"] > 0 else "deceptive"
    pn = "positive" if score["positive_negative"] > 0 else "negative"
    return td, pn


with open(path_to_model, "r", encoding="utf-8") as txt_reader:
    model = json.load(txt_reader)
    model["w"]["truthful_deceptive"] = np.array(model["w"]["truthful_deceptive"])
    model["w"]["positive_negative"] = np.array(model["w"]["positive_negative"])

test_file_path = get_all_file_path(path_to_test, 0)

output = ""
for test_file in test_file_path:
    with open(test_file, "r", encoding="utf-8") as file_reader:
        doc = ""
        for line in file_reader:
            doc += line.lower()
        sub_doc = multiple_replace(neg_dict, doc)
        td, pn = classify(sub_doc, model, ["truthful_deceptive", "positive_negative"])
        output += td + " " + pn + " " + test_file + "\n"


with open("percepoutput.txt", "w", encoding="utf-8") as txt_writer:
    txt_writer.write(output.strip())