# Grading
import sys

def split_word_tag_token(token):
    if len(token.split('/')) == 2:
        w = token.split('/')[0]
        t = token.split('/')[1]
    else:
        t = token.split('/')[-1]
        w = ''.join(token.split('/')[:-1])
    return w, t


path_to_result = sys.argv[1]
path_to_dev = sys.argv[2]
hmm_output = []
devset = []
with open(path_to_result, 'r') as reader:
    for line in reader:
        hmm_output.append(line)

with open(path_to_dev, 'r') as reader:
    for line in reader:
        devset.append(line)

if len(hmm_output) != len(devset):
    raise Exception("Number of lines is not equal. Exit with Error(1)")

ntags = 0
ncorrect = 0
for i in range(0, len(devset)):
    lhmm = hmm_output[i].split(' ')
    ldev = devset[i].split(' ')
    for j in range(0, len(ldev)):
        hmm_w, hmm_t = split_word_tag_token(lhmm[j])
        dw, dt = split_word_tag_token(ldev[j])
        if hmm_w != dw:
            raise Exception("Word does not match hmm_w={} dw={}. Exit with Error(1)".format(hmm_w, dw))
        if hmm_t == dt:
            ncorrect += 1
        ntags += 1

print("Correct: {}".format(ncorrect))
print("Total: {}".format(ntags))
print("Accuracy: {}%".format(ncorrect * 100 / ntags))