import sys
import json
import math


def split_word_tag_token(token):
    if len(token.split('/')) == 2:
        w = token.split('/')[0]
        t = token.split('/')[1]
    else:
        t = token.split('/')[-1]
        w = ''.join(token.split('/')[:-1])
    return w, t


def handle_unseen(T, idx, b):
    if T[idx] not in b.keys():
        b[T[idx]] = b['##unseen##']


def handle_no_possible_transition(Q, T, t, b, prob, back_pointer):
    max_prev_t = 2
    most_likely_pq = None
    for ipq, pq in enumerate(Q):
        if prob[ipq][t - 1] == 2:
            continue
        if max_prev_t == 2:
            max_prev_t = prob[ipq][t - 1]
            most_likely_pq = pq
        else:
            if prob[ipq][t - 1] > max_prev_t:
                max_prev_t = prob[ipq][t - 1]
                most_likely_pq = pq

    max_cur_emission = 2
    cq = None
    most_liekly_iq = -1
    for iq, q in enumerate(Q):
        if b[T[t]][q] == 0:
            continue
        if max_cur_emission == 2:
            max_cur_emission = math.log(b[T[t]][q])
            cq = q
            most_liekly_iq = iq
        else:
            if math.log(b[T[t]][q]) > max_cur_emission:
                max_cur_emission = math.log(b[T[t]][q])
                cq = q
                most_liekly_iq = iq
    back_pointer[cq][t] = most_likely_pq
    prob[most_liekly_iq][t] = max_prev_t + max_cur_emission


def viterbi(sent, hmm_model):
    # Viterbi
    back_pointer = {}
    T = sent.split(' ')
    Q = hmm_model['tag']
    a = hmm_model['p_transition']
    b = hmm_model['p_emission']
    q0 = '#Q0'
    qt = '#QT'
    num_q = len(Q)
    num_t = len(T)
    # Initialize by 1 which is INF here and will be skip for unreachable state
    prob = [[2 for t in range(num_t + 1)] for q in range(num_q)]  # num_t + 1 include last state QT

    # Note:
    # cal neg value --> find max arg --> skip no value
    # or negate to get pos value -> find min --> like find min entropy

    # TODO - Smoothing a[q'][q] == 0
    # initialize at step t_0
    for iq, q in enumerate(Q):
        back_pointer[q] = {}  # initialize backpointer with all key 'q' in Q
        # TODO - handle unseen word b[T[0]]
        handle_unseen(T, 0, b)
        if a[q0][q] == 0 or b[T[0]][q] == 0:
            continue
        prob[iq][0] = math.log(a[q0][q]) + math.log(b[T[0]][q])
        back_pointer[q][0] = q0
        # Handle no transition

    # Run from step t_1 to t_T
    for t in range(1, num_t):
        # find argmax prob
        argmax_b = 1
        has_transition = False
        for iq, q in enumerate(Q):
            for ipq, pq in enumerate(Q):  # Enumerate every q in previous t
                handle_unseen(T, t, b)
                if prob[ipq][t - 1] == 2 or a[pq][q] == 0 or b[T[t]][q] == 0:
                    continue
                if prob[iq][t] == 2:
                    prob[iq][t] = prob[ipq][t - 1] + math.log(a[pq][q]) + math.log(b[T[t]][q])
                    argmax_b = prob[ipq][t - 1] + math.log(a[pq][q])
                    back_pointer[q][t] = pq
                    has_transition = True
                else:
                    prob[iq][t] = max(prob[iq][t], prob[ipq][t - 1] + math.log(a[pq][q]) + math.log(b[T[t]][q]))
                    if prob[ipq][t - 1] + math.log(a[pq][q]) > argmax_b:
                        argmax_b = prob[ipq][t - 1] + math.log(a[pq][q])
                        back_pointer[q][t] = pq

        if not has_transition:
            # Prob: when state 't' is unreachable
            # case: there is only one transition to QT and that word has no b[w][pos] value
            # Or the only b[w][pos] we have is not reachable
            # Sol: use most likely tag for current 't' and use most likely tag for prev t
            handle_no_possible_transition(Q, T, t, b, prob, back_pointer)

    # At termination step: QT
    bpt_qt = None
    argmax_qt = 2
    for iq, q in enumerate(Q):
        # TODO - add smoothing when a[q][qt] == 0
        # Edge case: there is only one transition to QT and that word has no a[q][qt] value
        if prob[iq][num_t - 1] == 2 or a[q][qt] == 0:
            continue
        if argmax_qt == 2:
            argmax_qt = prob[iq][num_t - 1] + math.log(a[q][qt])
            bpt_qt = q
        else:
            if prob[iq][num_t - 1] + math.log(a[q][qt]) > prob[iq][num_t]:
                prob[iq][num_t] = max(prob[iq][num_t], math.log(a[q][qt]))
                bpt_qt = q

    # Return backtrack path
    q = bpt_qt
    sent_tag = []
    t = len(T) - 1
    while q != '#Q0' and t >= 0:
        sent_tag.append(q)
        q = back_pointer[q][t]
        t -= 1
    sent_tag.reverse()

    output = ""
    for i, token in enumerate(T):
        output += "{}/{} ".format(token, sent_tag[i])
    output = output.strip()
    return output


path_to_input = sys.argv[1]
with open('./hmmmodel.txt', 'r') as reader:
    hmm_model = json.load(reader)

sents = []
out_sents = []
with open(path_to_input, 'r') as reader:
    for line in reader:
        sents.append(line.strip())

for sent in sents:
    tagged_sent = viterbi(sent, hmm_model)
    out_sents.append(tagged_sent)

with open('./hmmoutput.txt', 'w') as writer:
    for tagged_sent in out_sents:
        writer.write(tagged_sent + "\n")




