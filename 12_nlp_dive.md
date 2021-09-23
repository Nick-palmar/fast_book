## A language model form scratch

1. If the dataset for your project is so big and complicated that it will take a significant amount of time to work with it, you should either simplify, reduce, or find another simpler dataset that will allow you as a practitioner to test your methods and interpret their results. The prototyping stage is meant to be quick and easy so the dataset should reflect this idea. 

2. Documents in the dataset are concatenated before creating the langauge model so the data can be layed out as a stream of continuous text (predicting the next token based on the prev token). Also, by concatenating the documents, you are ensuring that later on all input tensors will be of the same shape without mismatches (easier to make the batches). 

3. To predict the 4th word in the model given the first 3, here are the 2 tweaks needed to be made to a standard fully connected network (using 3 standard linear/dense layers):
- Each layer will take the the respective column from the embedding matrix (layer one will use the first word's embedding, layer 2 will use the 2nd word's embedding, etc). Each next layer will take the corresponding input word's embedding matrix as well as the previous layer's output to interpret information in context (as sequential data).
- Each layer will use the same weight matrix. This means that activations will change as new words move through layers, but the weights will stay the same (we do not want to learn words/tokens in specific positions, we just want to learn the words in general and handle words in all positions). 
