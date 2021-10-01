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

 13. Write code to print out the first few batches of the validation set, including converting the token IDs back into English strings, as we showed for batches of IMDb data in Chapter 10. Assume we start with seqs, a list of tensors, seqs, which is a list of tokenized and numericalized tensors

 ```
 def group_chunks(dset, bs):
    # get the length of a group 
    m = len(seqs)//bs
    new_dset = L()

    # loop through each possible value in m
    for i in range(m):
        # append (0, m, 2*m, ..., (bs-1)*m)
        # then increment i and append (1, m+1, 2*m+1, ..., (bs-1)*m+1) etc
        new_dset += L(dset[i+ m * j] for j in range(bs)0
    return new_dset


# check if the function works
print(seqs[0], seqs[1], seqs[2])
s_chunks = group_chunks(seqs, 64)
# should be the same as above
print(s_chunks[0], s_chunks[64], s_chunks[128])

train_cut = int(len(seqs) * 0.8)
bs = 64

dls = DataLoaders.from_dsets(group_chunks(seqs[:train_cut], bs), group_chunks(seqs[train_cut:], bs), bs=bs, drop_last = True, shuffle = True)

# print first few batches in validation set

for i, (x, y) in enumerate(dls.valid):
  if i > 3:
    break
  
  words = " ".join(vocab[num] for num in x[0])
  print(words)
 ```


 14. The ModelResetter callback will call the reset() method of the model at the beginning of each epochs as well as at the beginning of the validation phase. The reset method resets the hidden state, which ensures that the model's hidden state/memory resets when it starts to read a new continous chunk of text (ie new epoch or new validaiton set for text). 


 15. The downside of predicting just one word for every 3 inputs is that we are losing alot of our signal since we could be predicting the next word after the current word (for all words). This way, we have more 'training data' for the model to learn from (even though the amount of data is still technically the same). This can be done by simply changing the 

 16. LMModel4 requires a custom loss function because the multiple signals means that after every forward passed, the stacked tensor is of shape (bs, seq, vocab_sz). This does not match up with a target of shape (bs, seq). The target itself must be flattened to an individual value (recall cross entropy loss requires the target for indexing for the NLLLoss part). So, we flatten the target by using targ.view(-1). We must match up the input by doing inp.view(bs*sl, -1) or inp.view(-1, len(vocab)). This way, each input will provide a prediction for the next word in the vocab that can be indexed into by the single target value.


 17.  The training for LMModel4 is unstable because the neural network is very deep, which can result in very large or small gradients. These gradients result in bad step sizes and can complicate the training. 

 18. Stacked RNNs can help improve the results even though RNNs are already quite deep because a stacked RNN provides a different linear layer with a different weight matrix between the hidden state and output activations. This new linear layers makes the model more flexible as this layer can be optimized differently from the first layer by SGD. 

 19. 
 

  ![Unrolled stacked RNN](https://github.com/Nick-palmar/fastai_deep_learning/blob/main/images/rnn_unrolled.png?raw=true)

 ![Rolled stacked RNN](https://github.com/Nick-palmar/fastai_deep_learning/blob/main/images/rnn_rolled.png?raw=true)


 20. We should get better results if we call detach less often bc it lets the weights be optimized deeper into the model's history (longer time horizon to learn from). In practice with a simple RNN, this may not happen because if it is only a single layer RNN, then it will only have one linear layer that it can train. As a result, less features can be learned even if the model is able to take gradients further in the past (by calling detch less). 

 21. A deep network can result in very large or small activations because a deeper network must perform many multiplications (linear layers). If you multiply a number < 1 many time, the number will become very small (a very small activation in the network). Conversly, multplying a number > 1 many times will result in very large activations. This is problematic because very small numbers and very large numbers are not so simple to store in the computer. Really large numbers are also less percise in floating point repr. which causes problems in gradient calculations/steps in SGD (vanishing/exploding gradients). 

 22. In a computer's floaitng point, numbers close to 0 are the most precise. 

 23. Vanishing gradients prevent training bc when the gradients get too small, the steps in SGD will not update the parameters either. This means that the model is not being trained well. 


 24. The two hidden states in LSTM architecture help to delegate tasks between memory and output. The first state is called the cell state, and it's specific task is to retain long short term memory. The second state is called the hidden state, and that one will focus on predicting the next token in a sequence. 

 25. The two hidden states are called cell (ct) and hidden (ht) states. 

 26. tanh is another activation function and it is related to sigmoid as it is pretty much just sigmoid rescaled from -1 to 1. Mathematically speaking, 
 ```
 tanh(x) = 2*sigmoid * 2*x - 1
         = (e^x - e^-x) / (e^x = e^-x)
 ```

Intuatively, sigmoid is used to determine what values to turn on and off while tanh is used to determine the values (ie. the input and cell gates in the LSTM). 


27. 
```
h = torch.cat([h, input], dim=1)
```
In the LSTM cell, this code allows for inputs (the tokens after passing through the embedding layer) and hidden state to be concatenated before being passed to the gates in the LSTM. Since the embeddings and hidden states are being concatenated, this allows for both tensors to be of different shapes. The linear layers for the games in the LSTM NN go from (emb_sz + hidden_sz) -> (hidden_sz) after passing through the gates. 

This differs from previous RNNs explored in this chapter because normally, the first line of code was: self.h = self.h + self.i_h(self.h). The hidden size and embedding had to be the same shape since they were being added together. The new concatenation and passing them through the mini-networks (gates) allows for both inputs to the gate to be of differnet size. 

28. chunk in pytorch allows for a tensor to be split into pieces along a dimension. You specify the nuber if chunks to create along a dimension and pytorch will return the number chunks you requested as a list of tensors. 


29. Why the refactored version of LSTM is the same as non-refactored: The ih and hh are treating input and h separately as in the first LSTMCell. This part works slightly differently, but in effect they are treated separately then combined by addition once they both have 4*n_hidden as a dimension (after passing through their linear layer). This dimension is then chunked into 4 gates so that each gate has dimension n_hidden. The first 3 gates have sigmoid applied while the final cell gate has tanh applied. Then the same computations are applied to calculate h and c. 


30. We can use a higher learning rate to train LMModel6 (contains LSTM layer) because LSTMs avoid the exploding gradient problem; as such they have smaller gradients so step sizes can be larger without having to worry about losing precision due to inaccuracy of fp calculation for numbers far away from zero. Since param -= param.grad * lr, a smaller param.grad from the LSTM can be combined with a larger lr to step an LSTM (compared to a vanilla RNN with possily large param.grad so requiring smaller lr). 

31. The AWD-LSTM model must use the following regularizations techniques (by definition): dropout, activation regularization, and temporal activation regularization. 

32. Dropout is a regularization technique where some a set of random activations are turned to zero in training. Intuatively, dropout helps neurons cooperate better and makes the activations more nosiy allowing for better generalization. 

33. The activations are scaled with dropout because dropout drops a random activaitons and thus changes the magnitudes of the activations further down the model. For example, it would be problematic if instead of adding 5 activations in a model, only 2 were being added with dropout. Scaling allows for the activations in the model to remain of similar magnitude and continue to train well. 

Dropout in training removes neurons with probability p and scales them by dividing activations by (1-p). In inference, scaling can still occurs as weights are multiplied by the probability of dropout 1-p ((1-p)*w). Note that the scaling is applied either in scaling or in inference-  not both (either div by 1-p in training or mult by 1-p in inference)


34. The purpose of the line from dropout
```
if not self.training: return x
```
is to not dropout a layer if it is in inference/eval mode. When we are predicting things, we want to use all the neurons so not activations should be zeroed out if we are not training - as ensured by this line. 

35. Experiment with bernoulli
```
# experiment 1
tens = torch.abs(torch.randn((4, 2))).div(3)
bern_tens = torch.bernoulli(tens)
print(tens, bern_tens)

# experiment 2
shaped_tens = torch.randn((3, 3))
probs = torch.arange(start=0, end=1, step=0.1)
for prob in probs:
  print(f"Prob {round(prob.item(), 1)}: \n", shaped_tens.bernoulli(prob) , '\n\n')

# experiment 3
new_tens = tens.new_zeros(*tens.shape).bernoulli_(0.5)
print(new_tens)
```

36. In pytorch, so set model to be in training model do model.train() and in evaluation mode do model.eval()

37. The equation for activation regularization is: 
```
# add the means of the final layer of activations squared times a multiplier, alpha
loss += alpha * activations.pow(2).mean()
```
Activation regularization is different from weight decay because it is trying to make the final activations produced by the LSTM, not the parameters, as small as possible. Intuatively, it prevents the LSTM outputs/activations from overfitting. 

38. The equation for temporal activation regularization is
```
# assume activations are shape (bs, seq, n_hidden)
loss += beta * (activations[:, 1:] - activations[:, :-1]).pow(2).mean()
```
TAR is ensuring that the difference between consecutive activations is as small as possible (since words in a sequence should make sense, thus activations should not change so much as they are results of predicting tokens in a sequence). 
We would not use TAR for computer vision problems because subsequent activations do not align with sequenced data necessarily. Different parts of an image can be completely unrelated, so TAR would not make sense. 


39. Weight tying is the process of setting the output layer weights equal to the input (embedding) layer weights. The reason for this is that input is going from english -> hidden, while output is just going from hidden -> english in a language model. Intuatively, this mappings can be the same. 