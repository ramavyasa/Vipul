
# coding: utf-8

# In[69]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[70]:


data = open("ner.txt",'r').readlines()
data = [line.strip('\n').split(' ') for line in data]


# In[71]:


sentence_vector = []
llist = []
words = set()
tags = set()
for i in data:
    if i == ['']:
        sentence_vector.append(llist)
        llist = []
    else:
        tags.add(i[1])
        words.add(i[0])
        llist.append(tuple(i))


# In[72]:


print(sentence_vector[0])


# In[73]:


tags = list(tags)
words = list(words)


# In[74]:


n_words = len(words)


# In[75]:


n_tags = len(tags)


# In[76]:


max_len = 100
word2int = {w: i for i, w in enumerate(words)}
tag2int = {t: i for i, t in enumerate(tags)}


# In[77]:


#word2int["live"]


# In[78]:


#tag2int["D"]


# In[79]:


X = [[word2int[w[0]] for w in s] for s in sentence_vector]


# In[80]:


X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)


# In[81]:


#X[1]


# In[82]:


y = [[tag2int[w[1]] for w in s] for s in sentence_vector]


# In[83]:


y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2int["O"])


# In[84]:


y = [to_categorical(i, num_classes=n_tags) for i in y]


# In[85]:


X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
#print(len(X_tr))
#print(len(X_te))


# In[86]:


input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words, output_dim=100, input_length=max_len)(input)
#model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer


# In[87]:


model = Model(input, out)


# In[88]:


model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])


# In[89]:


history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=5, validation_split=0.1, verbose=1)


# In[90]:


hist = pd.DataFrame(history.history)


# In[91]:


plt.figure(figsize=(12,12))
plt.plot(hist["acc"])
plt.plot(hist["val_acc"])
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[92]:


i = 254
p = model.predict(np.array([X_te[i]]))
p = np.argmax(p, axis=-1)
print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
for w, pred in zip(X_te[i], p[0]):
    print("{:15}: {}".format(words[w], tags[pred]))

