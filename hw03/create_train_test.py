import os
import sys
from shutil import copyfile, copytree, rmtree, ignore_patterns

resource = "./resources"
if len(sys.argv) > 1:
    test_fold = "fold" + sys.argv[1]
else:
    test_fold = "fold1"

neg_dec = os.scandir("./resources/negative_polarity/deceptive_from_MTurk")
neg_tr = os.scandir("./resources/negative_polarity/truthful_from_Web")
pos_dec = os.scandir("./resources/positive_polarity/deceptive_from_MTurk")
pos_tr = os.scandir("./resources/positive_polarity/truthful_from_TripAdvisor")

dir_list = [neg_dec, neg_tr, pos_dec, pos_tr]

dst_dir = "./test_data"
ground_truth = ""
# for i, directory in enumerate(dir_list):
#     for fold in directory:
#         if fold.name == test_fold:
#             for file_data in os.scandir(fold):
#                 if i == 0:
#                     name_prefix = "nd_"
#                     ground_truth += "deceptive negative "
#                 elif i == 1:
#                     name_prefix = "nt_"
#                     ground_truth += "truthful negative "
#                 elif i == 2:
#                     name_prefix = "pd_"
#                     ground_truth += "deceptive positive "
#                 else: # 3
#                     name_prefix = "pt_"
#                     ground_truth += "truthful positive "
#                 dst_full_path = dst_dir + "/" + name_prefix + file_data.name
#                 ground_truth += dst_full_path + "\n"
#                 copyfile(file_data, dst_full_path)
dst_test = "./test_data"
copytree("./resources/negative_polarity", os.path.join(dst_test, "negative_polarity"), ignore=ignore_patterns('.DS_Store'))
copytree("./resources/positive_polarity", os.path.join(dst_test, "positive_polarity"), ignore=ignore_patterns('.DS_Store'))

dst_train = "./train_data"
copytree("./resources/negative_polarity", os.path.join(dst_train, "negative_polarity"), ignore=ignore_patterns('.DS_Store'))
copytree("./resources/positive_polarity", os.path.join(dst_train, "positive_polarity"), ignore=ignore_patterns('.DS_Store'))

# remove train fold
with os.scandir("./test_data/negative_polarity") as train_entries:
    for folder in train_entries:
        for fold in os.scandir(folder):
            if fold.name != test_fold:
                rmtree(fold.path)

with os.scandir("./test_data/positive_polarity") as train_entries:
    for folder in train_entries:
        for fold in os.scandir(folder):
            if fold.name != test_fold:
                rmtree(fold.path)

# remove test fold
with os.scandir("./train_data/negative_polarity") as train_entries:
    for folder in train_entries:
        for fold in os.scandir(folder):
            if fold.name == test_fold:
                rmtree(fold.path)

with os.scandir("./train_data/positive_polarity") as train_entries:
    for folder in train_entries:
        for fold in os.scandir(folder):
            if fold.name == test_fold:
                rmtree(fold.path)

# with open("ground_truth.txt", "w") as txt_writer:
#     txt_writer.write(ground_truth)

neg_dec.close()
neg_tr.close()
pos_dec.close()
pos_tr.close()