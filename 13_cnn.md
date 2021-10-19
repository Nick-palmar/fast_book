
## 13. CNNs

1. A feature is a change made to the data so that it becomes easier to model/represent generally

2. The conv kernel matrix for a top edge detector is:

```
top_edge = [
            [-1, -1, -1], 
            [0, 0, 0], 
            [1, 1, 1]
        ]       
```

Intuatively, this kernel penalizes high pixel values at the top of the kernel (mult by -1) and favours high pixel values at the bottom of the kernel. This means that the result of the convolution is high when the top layer is lower and the bottom layer is high (ie. you are moving from light to dark with a top edge).


3. The mathematical operation for a single kernel is done by multiplying the kernel by the same size of pixels around the target pixel and summing up the results. Mathematically with a 3x3 kernel, this is:

```
kernel = [[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]

pixel = (row, col)

row_1 = img[row-1, col-1]*a1 + img[row-1, col]*a2 + img[row-1, col+1]*a3

row_2 = img[row, col-1]*b1 + img[row, col]*b2 + img[row, col+1]*b3

row_3 = img[row+1, col-1]*c1 + img[row+1, col]*c2 + img[row+1, col+1]*c3

conv_res = row_1 + row_2 + row_3
```


4. Applying a convolutionalkernel to a 3x3 matrix of zeros is zero. This is bc matrix_val = 0 and you have convolution =  (kernel_val * matrix_val).sum(). Thus, kernel_val * matrix_val is always 0 yielding a result of zero. 

5. Padding are additional pixels added to the outside of an image. Padding helps to make the output of the convolutional layer the same size as the original image - this can be done by making padding be kernel_size // 2 assuming kernel_size is an odd number (does not work for even kernel_size). 


6. Stride is the number of pixels to slide the kernel after performing a convolution on a window in an image. It can be used to reduce the output size (if stride-2 convolutions are being used - otherwise if stride-1 convolutions are in use output size will remain the same). 


7. Use nested list comprehension to create an alphabet grid:

```
import string
alpha = string.ascii_lowercase

alpha_grid = [(char, letter) for letter in alpha for char in alpha]

```

8. On forward call, torch's conv2d operation takes the parameters input and weight. Since the operation is accelerated on the gpu, these are both 4-dim tensors. The input parameter is expected to be of shape (batch_size, in_channels, image_height, image_width), while the weight parameter is expected to be out shape (output_channels, in_channels, kernel_height, kernel_width). Notice that in both, the in_channel isat the same place as the 2nd dimension as these must match up - each image channel must have a corresponding kernel associated with it. A single one of these layers with multiple kernels being applied to the respective channels is considered a filter. 

9. A channel refers to the number of pixels/activations related a single grid location of a particular image. Channel in general can be applied to input images (RGB images have 3 channels which are the 3 colours bc each grid location is represented by 3 different pixels - RGB) and also activations (a convolutional layer can apply multiple kernels to the input channels such that each grid location ends up with a different number of activations, known as the output channels). 


10. A covolution is the result of applying a kernel across an image. The result is that the convolution can be thought of as a special type of matrix multiplication where the weight matrix is of shape (kernel_size_flat, input_size_flat) and the second input matrix is a vector of size (input_size_flat, 1). The weight matrix has two special properties: it has 0 weights which cannot be trained and some of the weights in the matrix are 'share weights' such that they can change by SGD but must keep the same value as they are updated. The below image shows a convolutional operation in matrix multiplication form:


 ![MM Convolution](https://github.com/Nick-palmar/fastai_deep_learning/blob/main/images/mm_conv.png?raw=true)



 11. A convolutional neural network (or CNN) is a neural network that uses convolutional layers (and can also contain dense layers). These convolutional layers contain weight matrices of kernels that are updated by SGD like in other neural networks to optimize the features picked up by the kernels (ie. no kernel is preferred over other kernels). 


 12. The benefits of refactoring parts of your NN definition are that it is less likely you will make mistakes since the same refactoring can be used in multiple places and it makes it easier for others to see which parts of layers are being changed (since a separate refactoring is occuring rather than just embedding new code in a large space). 

 13. Flatten is a layer in a neural network that performs squeeze operations on the tensors in module form. For example, a tensor of shape (bs, channels, 1, 1) can be changed to (bs, channels) by using a flatten layer. In the MNIST CNN, flatten needs to be included as the final layer so that the output tensors are of shape (bs, categories) to match up with the expections of the loss function (cross entropy expects 2 categories). A (batch_size, categories, 1, 1) tensor would be problematic for cross entropy loss and would likely cause errors - flattening fixes this. 


 14. NCHW refers to the axes on tensors in the neural network as procesed by pytorch. It stands for (N, C, H, W) or (batch_size, channel, image_height, image_width). Conversely, tensorflow uses NHWC ordering for tensors (ie. channel as the final axis). 


 15. The third layer of MNSIT CNN has 7x7x(1168-16) multiplicaitons because the inputs have shape bsx8x7x7, outputs have shape bsx16x4x4 and the layer itself has 1168 parameters. The convolution is doing multiplication at each pixel in the 7x7 input tensor (a single kernel centered at each pixel in the 7x7 input space = a single location) and the number of weights gives the number of multiplications per location, yielding 7x7xweight_size multiplications. The weight_size is 1168-16 because 16 of the parameters are bias terms (shown by the 16 in the channel axis of the output tensor) so the remaining parameters are the weights which each have a multiplication to perform at a location (ie. 7x7xweight_size multiplications). 


 16. A receptive field is the area in an image the is responsable for the calculation of a layer. Intuatively speaking, the deeper into a network you go, the larger the receptive field for a single activation in a layer (assuming each kernel is 3x3, each layer will take a 3x3 region and compress it so a single value and do this multiple times; the deeper in a network you are the more this has occured so the larger the receptive field of that specific activation). 


 17. Assuming a kernel size of ks and stride=2 (stride-2 convolutions), going from 2nd layer -> 1st layer requires a ksxks area from the 1st layer. Now, going from 1st layer -> input, each activation in the ksxks area of the 1st layer must come from a ksxks area in the input layer. Since we have stride-2 convolutions, we have (ks+2+2)x(ks+2+2) or (ks+4)x(ks+4) receptive field size for the activation. 


 18. Completed outside of this file

 19. Completed outside of this file

 20. A colour image is represented as a rank 3 tensor with 3 colours channels as the first axis (RGB) and the other 2 axes being image height/width.  

 21. A convolution with a colour input works by using a tensor of kernels (called a filter) rather than a single kernel. Each channel (3 RGB channels for images) will have a corresponding kernel and said kernel will be applied to each channel. To finish off the convolution for an RBG input, the results from the kernel from each channel will be added activation-wise to yield a single result of activation_sz*activation_sz for a single filter (ie. kernel_out_red + kernel_out_blue + kernel_out_green + bias). This will occur for channel_out channels, forming a channel_out * activation_sz * activation_sz output. The weights for the layer will be of size channel_out * channel_in * ks * ks. 


 22. After creating Dataloaders, to see the images one can use:

```
dls.show_batch()
```

23. Using stride-2 convolutions with padding of size ks//2 will half the resulting activation size of the output tensors. For example, if the input tensors to a convolutional layer had the HW dimensions as 28x28, stride-2 convolutions will reduce the activations to be 14x14. From previous knowledge, I know that later layers build up more complex features (richer features with larger receptive fields) so it makes sense to keep the capacity of the model for the model to be able to learn these complex features in later layers. In order to keep the capacity in terms of parameters and computation, the number of filters are doubled. 


24. We use a larger kernel in the first conv with MNIST because initially we are going from 1 channel -> 4 channels, then from 4 channels -> 8 channels. When going from 4 channels to 8 channels with a 3x3 kernel, we have 8 activations (output activations) for a grid segment (input activations) of size 9 (3x3). Neural networks are really only effective at creating useful features if they are foced to do so - that is the number of outputs is quite a bit smaller than the number of inputs. To overcome this, we change the first kernel size to 5x5 so that 25 grid input activations going to 8 output grid activations. 


25. ActivationStats saves the mean, standard deviation, and histogram of activations for every layer can can be trained (ie. requires grad). 

26. In fastai, a learner's callback after training can be accessed via the callback's name in snake_case. So the ActivationStats callback (which is a class) can be accessed through learn.activation_stats. 


27. plot_layer_stats from activation callback plots the mean, standard deviations, and percentage of weights near zero for a specific layer in the network. The xaxis represents the number of batches while training, while the yaxis has the corresponding statistic. 

28. Activations near zero are problematic because they are useless computations (model is doing nothing; anything*0 = 0) and will cause subsequent multiplications to be zero as well. This wil propagate to the final activation layers of the model, where many zero multiplications from previous layers will carry through even more making for lots of information loss. 


29. The upside of training with a larger batch is that the data will be less nosiy bc it will have more data to compute more accurate gradients on and thus be more representative of the training data. The downside is that there are fewer batches per epoch so less opportunities for the model to update the weights. 


30. We should avoid using high learning rates near the start of training to avoid diverging (ie. moving away from the absolute minimum) due to bad starting weights coupled with a high learning rate. 


31. 1cycle training is a form of changing the learning rate and momentum of the optimizer to train faster and more accurately (developed by leslie smith). In terms of learning rate, 1cycle training starts training with a low learning rate to avoid divergence, then moves to a higher learning rate in the middle of training (to generalize better by bypassing any local minimums and train faster) and finally comes back down to a low learning at the end to get the deepest possible into the final minimum. Momentum works opposite to the learning rate in that higher learning rate has a lower momentum coupled with it (and vice versa). The 1cycle training policy consists of a warmup phase (from low lr -> max lr) and an annealing phase (from max lr -> low lr). 


32. The benfits of training with a high learning rate are that you can bypass local minimums in training better with a higher learning rate leading to better generalizations. High learning rates means that the steps will jump around alot so it will attempt to find a point in the loss function that is generally stable and will not change too much with step sizes in it's general area. It also speeds up the training process since larger LR find the minimums faster than smaller LRs. 



33. Low LR near the end of training (annealing phase of 1cycle training) is good because after we have found a general local minimum using a high LR, we would like to lower the learning rate to find the lowest point in this general minimum section found from the high learning rate. It is like adding a finishing touch to the general solution to get the best results possible. 


34. Cyclical momentum is the idea that momentum (considering previous steps when computing the gradient) should vary oppositely to LR in the 1cycle policy. As such, momentum goes from high -> low -> high (opposite to LR). 


35. The Recorder callback (accessed through learn.recorder) saves information on loss, metrics, and hyperparameters in training. 



36. One column of pixels in the 'color_dim' plot representation shows a histogram of the activation values of a single batch. The entire graph shows all of the histograms of all of the batches stacked up (left being batch 0 and right being the final batch). 


37. Bad training in 'color_dim' looks like waves moving up and down. This is because he non-zero weights appear to grow exponentially and then they crash down back to zero (appearing like a wave). This is not useful because the activations go back to the start so the training from before seems useless. Good training would be represented by a smooth curve that gradually increased the non-zero activations indicating that the model always continued to learn more with meaningful activations in a smooth manner.  Bad training can result in poor training and bad results (lots of zero activations are not good as shown in Q28). 


38. A batch norm layer contains beta and gamma trainable parameters; after normalizing the activations of a layer using the mean and standard deviation from a batch, the normalized activation vector for the layer will be adjusted by gamma*norm_activation_layer + beta. This allows for layers to be treated separetely from eachother (since the trainable parameters will combat against the effects of batch norm peformed in previous layers). These parameters mainly help to prevent the network from having really high activations to make accurate predictions. 


39. In training, the batch's mean and standard deviation are used to normalize a layer's activations. In validation, a running mean of the mean and standard deviation calculated during training are used.


40. Models with batchnorm generalize better because batch normalization adds an element of randomness to the training process. In training, the batches that are put together are slightly random and thus the mean/standard deviation are slightly random as well. This will change the amount of normalization each time. To make accurate predictions, the model must become robust to these changes and not fluctuate. 