import torch

infersent = torch.load('encoder/infersent.allnli.pickle', map_location=lambda storage, loc: storage)
infersent.use_cuda = False

glove_path = 'dataset/GloVe/glove.840B.300d.txt'

infersent.set_glove_path(glove_path)


sentences = [u"We will unveil details of the product next year.",
             u"We'll now turn to Kathy from Barclays.",
             u"Thank you for listening to us. Talk to you soon.",
             u"We anticipate a revenue of 3 billion in the next year.",
             u"Next quarter, we expect to release 3 billion in dividends to investors.",
             u"Well thank you very much and I want to add one more thing please.",
             u"Operator. It is you turn. Your mic is open",
             u"We will have more to say on that during our event next month."]
sentenceLabels = ["red", "green", "green", "red", "red", "blue", "green", "red"]

infersent.build_vocab(sentences, tokenize=True)

sentsVecs = infersent.encode(sentences, tokenize=True)



from math import*

def square_rooted(x):

    return round(sqrt(sum([a*a for a in x])),3)

def cosine_similarity(x,y):

    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)




sentencesWithVecs = []

for i, sentence in enumerate(sentences):
    sentencesWithVecs.append((sentence, sentsVecs[i]))

chosenSentenceIdx = 6
chosenSentence = sentencesWithVecs[chosenSentenceIdx][0]
chosenSentenceVec = sentencesWithVecs[chosenSentenceIdx][1]
print "Most similar sentences to:   {}\n\n".format(chosenSentence)

cosSimToSent = []
for sentTuple in sentencesWithVecs:
    sent = sentTuple[0]
    sentVec = sentTuple[1]
    sim = cosine_similarity(chosenSentenceVec, sentVec)
    cosSimToSent.append((sent, sim))
cosSimToSent.sort(key=lambda tup: tup[1], reverse=True)
print "\n".join("{}:  {}".format(sim, sent) for sent, sim in cosSimToSent)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X_pca = PCA().fit_transform(sentsVecs)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=sentenceLabels)
