# benchmark-for-Text-Classification
----
+ A simple benchmark for text classification


Requirement
----
+ pytorch
+ numpy
+ sklearn
+ nltk


Model
----
+ Single Layer Perceptron (MLP)
+ Convolutional Neural Network (CNN)
+ Multi-Head Attention (MultiHeadAttn)


Dataset
----
+ Keras.imdb
+ rt-polarity


Result
----
| Accuracy      | MLP           | CNN           | MultiHeadAttn |
| ------------- | ------------- | ------------- | ------------- |
| imdb          | 0.8658        | 0.82792       | 0.78408       |
| rt-polarity   | 0.7378        | 0.65759       | 0.64446       |


Usage
----
```
python RUN_imdb.py
```
or
```
python RUN_rt.py
```
