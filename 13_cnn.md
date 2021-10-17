
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