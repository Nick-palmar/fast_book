## NLP Deep Dive: RNNs

1. Self-supervised learning means not giving labels to a model - just giving it lots and lots of data and the model learning from the data by creating labels automatically. In NLP applications, models are often given lots of words and the labels are simply the next word after the current word. The labels are embedded in the independant variable. 

2. A language model is a model trained to guess the next word in text after reading the previous ones. 

3. A language model is considered self-supervised because it takes lots and lots of text data and creates the labels by offsetting the words by 1 in the independent variable. This way, the labels are always the next word in the sentence which are being predicted. 

4. Self-supervised learning is often used for pretraining of models using for transfer learning - it is not normally the 'final' model. 

5. Language models are fine tuned to a particular copus before adding a classifer layer at the end because the ULMFiT paper approach proved that fine tuning the langauge model prior to fine tuning the classificaiton model (end result) yielded better results. Intuatively, fine tuning the language model on a particular corpus makes the model understand the specific type of language better before converting it into a classifier of some type - as such, it has learned about the terminology associated with the classification task and will be more equipped to complete the classification task well. 

6. To create a state-of-the-art text classifier, one can follow the ULMFiT approach which has three steps: first training a language model on a large dataset (wikitext 103), then training the model on context specific data, and finally training a classifier with the context specific data. Jeremy Howard shows how this novel idea to nlp transfer learning is very effective with the addition of disciminative fine tuning, slanted learning rates. 

7. The 50,000 unlabelled movie reviews help create a better text classifier as they can be used to train the langauge model in the 2nd training step of ULMFiT (by training the language model on the 50,000 unlabelled movie reviews). As such, the language model will become familiar with the movie review corpus and generalize better as a classifier. The model will be very good at predicting the next words in a movie review. 