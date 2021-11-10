## 14. Resnets 

1. We got to a single vector of activations in the previous chapter (CNNs) by using enough stride-2 convolutions to decrease the activation size to 1x1 and then flattening those final activaitons. This is not suitable for imagnette because this requires too many stride-2 convolutions (probably more than we may want) and the architectures cannot be used on different sized images since they only apply to images where a certain number of stride-2 convolutions can leave their dimensions as 1x1. 

2. Instead of using stride-2 convolutions for imagenette, we use an average pooling layer which takes a grid of activations and replaces it by the average activation in that grid. This reduces the final grid of activations from nxn (after applying conv layers which may have different strides) to 1x1, which can then  be flattened as before. This is called a **fully convolutional network**. 

3. Adaptive pooling is the idea that we can specify a pool output size and the parameters of the pooling function can be chosen in such a way that meets this output size (the pooling function parameters are **adaptive**). 

4. Average pooling means to take the average over a grid of activations and replace the grid of activations by this average. 

5. An adaptive average pooling layer will leave behind unit axes (in the case of nn.AdaptiveAvgPool2d(1)) so Flatten() can be used to get rid of the 1x1 axes at the end after the pooling layer. 

6. 

5. 
