import json
import sys

path_to_input = sys.argv[1]

sents = []
with open(path_to_input, 'r') as reader:
    for line in reader:
        sents.append(line.strip())

# transition prob - pairwise prob of tag
p_transition = {}
transition_count = {}
# emission prob (#word-in-tag / #vocab-in-tag)
# P(observation|state) or P(word|tag) --> This version is better than a more intuitive version
# P(tag|word) ref. Charniak paper
p_emission = {}
tag_count = {}
word_count = {}


# count transition using sliding window
def split_word_tag_token(token):
    if len(token.split('/')) == 2:
        w = token.split('/')[0]
        t = token.split('/')[1]
    else:
        t = token.split('/')[-1]
        w = ''.join(token.split('/')[:-1])
    return w, t


for sent in sents:
    word_tag = sent.split(' ')
    for wt_token in word_tag:
        w, t = split_word_tag_token(wt_token)
        if t not in tag_count.keys():
            tag_count[t] = 0
        tag_count[t] += 1

        if w not in word_count.keys():
            word_count[w] = {}

        if t not in word_count[w].keys():
            word_count[w][t] = 0

        word_count[w][t] += 1

for sent in sents:
    mod_sent = ('#/#Q0 ' + sent + ' #/#QT').split(' ')
    i = 0
    while i < len(mod_sent) - 1:
        _, t1 = split_word_tag_token(mod_sent[i])
        _, t2 = split_word_tag_token(mod_sent[i + 1])

        if t1 not in transition_count.keys():
            transition_count[t1] = {}

        if t2 not in transition_count[t1].keys():
            transition_count[t1][t2] = 0

        transition_count[t1][t2] += 1
        i += 1

tag_list = list(tag_count) + ['#Q0', '#QT']
p_transition = dict.fromkeys(tag_list, {})
p_emission = dict.fromkeys(word_count.keys(), {})

for k in tag_list:
    p_transition[k] = dict.fromkeys(tag_list, 0)

for w in word_count.keys():
    p_emission[w] = dict.fromkeys(tag_list, 0)

for t1, next_states in transition_count.items():
    count = 0
    possible_tags = []
    for next_tag, c in next_states.items():
        possible_tags.append(next_tag)
        count += c

    for t2 in possible_tags:
        p_transition[t1][t2] = transition_count[t1][t2] / count

for w, poss_tag_count in word_count.items():
    for t, n in poss_tag_count.items():
        p_emission[w][t] = n / tag_count[t]

# Generate Possibility for unseen
ntags = sum([v for k, v in tag_count.items()])
l_tags = [(k, v / ntags) for k, v in tag_count.items()]
# l_tags = sorted(l_tags,key=lambda x: x[1], reverse=True)
p_emission['##unseen##'] = {}
for t, prob in l_tags:
    p_emission['##unseen##'][t] = prob
p_emission['##unseen##']

hmm_model = {}
hmm_model['p_transition'] = p_transition
hmm_model['p_emission'] = p_emission
hmm_model['tag'] = list(tag_count.keys())

with open('hmmmodel.txt', 'w') as writer:
    json.dump(hmm_model, writer)
