
from nltk.translate import bleu_score

def consensus(captions):

    max = 0.0
    consensus = captions[0]

    # get the max bleu_score
    for caption in captions:
        references = captions[:]
        references.remove(caption)
        socre = bleu_score.sentence_bleu(references, caption)
        if len(caption) > max:
            max = socre
            consensus = caption

    return consensus

def accuracy(predict, real):
    
    accuracy = 0
    for i, pre in enumerate(predict):
        references = real[i]
        score = bleu_score.sentence_bleu(references, pre)
        accuracy += score

    return accuracy/len(predict)

def accuracy_rnn(predict, real):
    
    accuracy = 0
    for r in real:
        score = bleu_score.sentence_bleu(r, predict)
        accuracy += score

    return accuracy/len(real)
