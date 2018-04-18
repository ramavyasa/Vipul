
# coding: utf-8

# In[200]:


import numpy
import nltk
from nltk.tokenize import word_tokenize


# In[201]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[202]:


from itertools import chain
import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


# In[203]:


#nltk.corpus.conll2002.fileids()
data = open("ner.txt",'r').readlines()
data = [i.strip('\n').split(' ') for i in data]


# In[204]:


get_ipython().run_cell_magic(u'time', u'', u"# train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))\n# test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))\nsentences = []\nl = []\nwords = set()\ntags = set()\nfor i in data:\n    if i == ['']:\n        sentences.append(l)\n        l = []\n    else:\n        tags.add(i[1])\n        words.add(i[0])\n        text = word_tokenize(i[0])\n        k = [i[0],nltk.pos_tag(text)[0][1],i[1]]\n        l.append(tuple(k))")


# In[205]:


count_O=0
count_D=0
count_T=0
for j in sentences:
    for i in j:
        if i[2]=='O':
            count_O = count_O + 1
        elif i[2]=='D':
            count_D = count_D + 1
        else:
            count_T= count_T + 1
print("The count of 'O' tags in the data is", (count_O))
print("The count of 'D' tags in the data is",(count_D))
print("The count of 'T' tags in the data is",(count_T))


# In[206]:


train_sents = sentences[0:int(0.8*len(sentences))]
#dev_sents = sentences[int(0.7*len(sentences)):int(0.8*len(sentences))]
test_sents = sentences[int(0.8*len(sentences)):]


# In[207]:


print("A sample POS tagged sentence is as follows-")
print(train_sents[0])


# In[208]:


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],        
    }
    if i > 1:
        word1 = sent[i-2][0]
        postag1 = sent[i-2][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-2:word.lower()': word1.lower(),
            '-2:word.istitle()': word1.istitle(),
            '-2:word.isupper()': word1.isupper(),
            '-2:postag': postag1,
            '-2:postag[:2]': postag1[:2],
            
        })
    else:
        features['BOS'] = True
        
    if i < len(sent)-2:
        word1 = sent[i+2][0]
        postag1 = sent[i+2][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+2:word.lower()': word1.lower(),
            '+2:word.istitle()': word1.istitle(),
            '+2:word.isupper()': word1.isupper(),
            '+2:postag': postag1,
            '+2:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


# In[209]:


get_ipython().run_cell_magic(u'time', u'', u'X_train = [sent2features(s) for s in train_sents]\ny_train = [sent2labels(s) for s in train_sents]\n\nX_dev = [sent2features(s) for s in dev_sents]\ny_dev = [sent2labels(s) for s in dev_sents]\n\nX_test = [sent2features(s) for s in test_sents]\ny_test = [sent2labels(s) for s in test_sents]')


# In[210]:


get_ipython().run_cell_magic(u'time', u'', u"crf = sklearn_crfsuite.CRF(\n    algorithm='lbfgs', \n    c1=0.1, \n    c2=0.1, \n    max_iterations=1000, \n    all_possible_transitions=True\n)\ncrf.fit(X_train, y_train)")


# In[211]:


labels = list(crf.classes_)


# In[212]:


y_pred = crf.predict(X_test)
acc=metrics.flat_f1_score(y_test, y_pred, 
                      average='weighted', labels=labels)
print("The f1-score on the test data is",(acc))


# In[213]:


# group B and I results
sorted_labels = sorted(
    labels, 
    key=lambda name: (name[1:], name[0])
)
print("The table for scores on test data is as follows-")
print(metrics.flat_classification_report(
    y_test, y_pred1, labels=sorted_labels, digits=3
))

