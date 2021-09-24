## A language model form scratch

1. If the dataset for your project is so big and complicated that it will take a significant amount of time to work with it, you should either simplify, reduce, or find another simpler dataset that will allow you as a practitioner to test your methods and interpret their results. The prototyping stage is meant to be quick and easy so the dataset should reflect this idea. 

2. Documents in the dataset are concatenated before creating the langauge model so the data can be layed out as a stream of continuous text (predicting the next token based on the prev token). Also, by concatenating the documents, you are ensuring that later on all input tensors will be of the same shape without mismatches (easier to make the batches). 

3. To predict the 4th word in the model given the first 3, here are the 2 tweaks needed to be made to a standard fully connected network (using 3 standard linear/dense layers):
- Each layer will take the the respective column from the embedding matrix (layer one will use the first word's embedding, layer 2 will use the 2nd word's embedding, etc). Each next layer will take the corresponding input word's embedding matrix as well as the previous layer's output to interpret information in context (as sequential data).
- Each layer will use the same weight matrix. This means that activations will change as new words move through layers, but the weights will stay the same (we do not want to learn words/tokens in specific positions, we just want to learn the words in general and handle words in all positions).

4. We share a weight matrix across multiple layers in pytorch by creating a single layer and applying it multiple times (ie. the recurrent part; the layer is repeated in a loop). 

5. Module to predict the third word given prev 2:
```
class SimpleLM(nn.Module):
    def __init__(self, vocab_sz, hidden_sz):
        # define embedding layer, hidden layer, and output layer 

        self.i_h = nn.Embedding(vocab_sz, hidden_sz)
        self.h_h = nn.Linear(hidden_sz, hidden_sz)
        self.h_o = nn.Linear(hidden_sz, vocab_sz)

        self.relu = nn.ReLU()
    
    def forward(x):
        # given x as a 2 word input
        h = self.relu(self.h_h(self.i_h(x[:, 0])))
        # add embedding to prev output, tweak #1 q3
        h += self.i_h(x[:, 1])
        # apply same hidden layer, tweak #2 
        h = self.relu(self.h_h(h))

        # predict the 3rd word with an output layer
        return self.h_o(h)

```



6. A recurrent neural network is a neural network that uses a loop to feed previous 'hidden state'/activations to the 'next' layer. It is a refactoring of a multi-layer NN with a loop. Example of SimpleLMRNN
```
class SimpleLMRNN(nn.Module):
    def _init_(self, vocab_sz, hidden_sz):
        # same layers as before

        self.i_h = nn.Embedding(vocab_sz, hidden_sz)
        self.h_h = nn.Linear(hidden_sz, hidden_sz)
        self.h_o = nn.Linear(hidden_sz, vocab_sz)

        self.relu = nn.ReLU()
    
    def forward(x):
        # given x as a 2 word input
        # note that it is problematic to initialize hidden state to zero here
        h = 0
        for i in range(2):
            h += self.i_h(x[:, i])
            h = self.relu(self.h_h(h))

        # predict the 3rd word with an output layer
        return self.h_o(h)

```


7. Hidden state is the activation/set of activations that are updated at each step of the RNN

8. In *LMModel1*, h is the equivalent of hidden state. Although it is not an rnn, this is the value that is constantly being updated after the embedding and h_h layers.

9. To maintan state in an RNN, it is important to pass the text to the model in order because the hidden state should learn from sequence to sequence about the context. If the text is not passed in order, then the hidden state will pick up on the context incorrectly between sequences. 

10. An 'unrolled' representation of an RNN is the layout of the RNN before refactoring it with the for loop (all of the repeated intermediate layers are written/draw multiple times with different inputs). 

11. Maintaining the hidden state in an Rnn can lead to memory and performance problems bc by initializing the hidden state in __init__, you end up making the model as deep as the number of tokens your train on. In each loop, the hidden state is updated so for back propagation, the model needs to go all the way back to the first time it was updated in the very first token. The fix is to call .detch() which removes gradient history from pytorch (not backpropagate all the way to the first token). 

12. BPTT (or backpropagation through time) is treating a RNN with one layer per time (refactored using a loop) as a single model and calculating gradients as usual (unroll network and calc gradients through each layer, then rolling the network back up and stepping the weights, since the weight matrix is the same for each layer). When we call .detch(), we are doing truncated BPTT. 

 ![BPTT](https://github.com/Nick-palmar/fastai_deep_learning/blob/main/images/bptt.png?raw=true)

 ![BPTT and TBPTT](https://github.com/Nick-palmar/fastai_deep_learning/blob/main/images/bptt_tbptt.png?raw=true)