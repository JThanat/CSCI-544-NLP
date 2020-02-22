import os
import re

stat_header = ["TP", "TN", "FP", "FN", "accuracy", "precision", "recall", "n", "f1"]
deceptive_stat = dict.fromkeys(stat_header, 0)
truthful_stat = dict.fromkeys(stat_header, 0)
positive_stat = dict.fromkeys(stat_header, 0)
negative_stat = dict.fromkeys(stat_header, 0)


ground_truth_dict = {}

with open("nboutput.txt", "r") as output_reader:
    for line in output_reader:
        l = line.split(" ")
        is_deceptive = "deceptive" in l[2]
        is_truthful = "truthful" in l[2]
        is_pos = "positive" in l[2]
        is_neg = "negative" in l[2]

        if is_truthful and is_deceptive:
            raise ValueError("value is both truthful and deceptive: {}".format(l[2]))
        if is_pos and is_neg:
            raise ValueError("value is both positive and negative: {}".format(l[2]))

        # deceptive truthful
        if is_truthful and l[0] == "truthful":
            truthful_stat["TP"] += 1
            deceptive_stat["TN"] += 1
        elif is_truthful and l[0] == "deceptive":
            truthful_stat["FP"] += 1
            deceptive_stat["FN"] += 1
        elif is_deceptive and l[0] == "deceptive":
            truthful_stat["TN"] += 1
            deceptive_stat["TP"] += 1
        elif is_deceptive and l[0] == "truthful":
            truthful_stat["FN"] += 1
            deceptive_stat["FP"] += 1

        # positive negative
        if is_pos and l[1] == "positive":
            positive_stat["TP"] += 1
            negative_stat["TN"] += 1
        elif is_pos and l[1] == "negative":
            positive_stat["FP"] += 1
            negative_stat["FN"] += 1
        elif is_neg and l[1] == "negative":
            positive_stat["TN"] += 1
            negative_stat["TP"] += 1
        elif is_neg and l[1] == "positive":
            positive_stat["FN"] += 1
            negative_stat["FP"] += 1


def cal_stat(stat):
    # Ref: https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/
    stat["n"] = stat["TP"] + stat["TN"] + stat["FP"] + stat["FN"]
    # Accuracy = TP+TN/TP+FP+FN+TN
    stat["accuracy"] = (stat["TP"] + stat["TN"])/(stat["n"])
    # Precision = TP/TP+FP
    stat["precision"] = (stat["TP"])/(stat["TP"] + stat["FP"])
    # Recall = TP/TP+FN
    stat["recall"] = (stat["TP"])/(stat["TP"] + stat["FN"])
    # F1 Score = 2*(Recall * Precision) / (Recall + Precision)
    stat["f1"] = 2*(stat["recall"] * stat["precision"])/(stat["recall"] + stat["precision"])


def print_stat(_class, stat):
    return "{} {:.2f} {:.2f} {:.2f}".format(_class, stat["recall"], stat["precision"], stat["f1"])


cal_stat(truthful_stat)
cal_stat(deceptive_stat)
cal_stat(positive_stat)
cal_stat(negative_stat)

dc = print_stat("deceptive", deceptive_stat)
tf = print_stat("truthful", truthful_stat)
ng = print_stat("negative", negative_stat)
ps = print_stat("positive", positive_stat)
f1 = "Mean F1: {:.4f}".format((deceptive_stat["f1"] + truthful_stat["f1"] + negative_stat["f1"] + positive_stat["f1"])/4)

output = 50*"#" + "\n"
output += "Result: recall | precision | f1 \n"
output += dc + "\n"
output += tf + "\n"
output += ng + "\n"
output += ps + "\n"
output += f1 + "\n"
output += 50*"#" + "\n"
print(output)
with open("result.txt", "a") as result_writer:
    result_writer.write(output)

# Results 1:
# deceptive 0.84 0.87 0.85
# truthful 0.86 0.83 0.85
# negative 0.96 0.85 0.90
# positive 0.87 0.96 0.91
# Mean F1: 0.8779