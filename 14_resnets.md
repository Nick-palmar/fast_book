## 14. Resnets 

1. We got to a single vector of activations in the previous chapter (CNNs) by using enough stride-2 convolutions to decrease the activation size to 1x1 and then flattening those final activaitons. This is not suitable for imagnette because this requires too many stride-2 convolutions (probably more than we may want) and the architectures cannot be used on different sized images since they only apply to images where a certain number of stride-2 convolutions can leave their dimensions as 1x1. 

2. Instead of using stride-2 convolutions for imagenette, we use an average pooling layer which takes a grid of activations and replaces it by the average activation in that grid. This reduces the final grid of activations from nxn (after applying conv layers which may have different strides) to 1x1, which can then  be flattened as before. This is called a **fully convolutional network**. 

3. Adaptive pooling is the idea that we can specify a pool output size and the parameters of the pooling function can be chosen in such a way that meets this output size (the pooling function parameters are **adaptive**). 

4. Average pooling means to take the average over a grid of activations and replace the grid of activations by this average. 

5. An adaptive average pooling layer will leave behind unit axes (in the case of nn.AdaptiveAvgPool2d(1)) so Flatten() can be used to get rid of the 1x1 axes at the end after the pooling layer. 

6. A skip connection is the part in the resnet block that does (x +) in the (x + conv2(conv1(x)) = x + F(x)). It provides a directe route to the output y,in the form y=x+F(x). As such, if we set F(x)=0 (set the gamma to zero in batch norm), then y=x. This is why it is a 'skip connection'. It is also sometimes called the identity branch. 

7. Skip connections allow us to train deeper models by simplifying the training process with a shallower network that works (final layers simly act as an identity) and then fine tuning the final layers to improve the results. This ability to take on shallower, better networks and then fine tune deeper networks through improved SGD is why skip connections are advantageous. In practice, many resnet blocks with skip connections are trained through SGD and these skip connections are what simplify the deeper training. 

8. Figure 14-1 shows that deeper networks train and perform worse than shallower networks on certain tasks. This led to the idea of skip connections because the thought is that any deep network should be able to perform as least as well as any shallower network (simply by setting the additional layers to the identity mapping). This means that the output must have access to the input in some way - similar to the skip connection. Additionally, if the gamma batch norm parameter is set to 0 in the final layers, you end up with y = x + F(x) = x, so you get that the final layers can take on the value of the shallower layers while still having trainable parameter to improve. 

9. Identity mapping is just returning the input without changing it at all. In the context of a resnet block, it would be y = x + F(x) where F(x) = 0 since F(x) = conv2(conv1(x)) and conv2 has batch norm at the end with gamma = 0. 


10. Basic equation for resnet block is x + conv2(conv1(x))

11. Resnets are performing the computation y = x + F(x). As such, this F(x) = y - x; it is trying to minimize the error between x and the desired y. Intuatively, this means that the resnet is good at learning difference between doing nothing (x) and passing through two convolutional layers. The residual part of resnets is the fact that they are built on the residual mapping F(x) = y - x = H(x) - x where F(x) is a new constructed mapping (2 conv layers) and H(x) is the desired underlying mapping. Rather than stacking layers on eachother, finding the residual function is much easier to do, hence F(x) = H(x) - x and we want H(x) = x + F(x), which is the formula for a resnet block. 


12. We deal with skip connections (x +) when there is a stride-2 convolution by applying a 2x2 average pooling layer to reduce the original x to be the same size as the activation grid after the stride-2 convolution. When the number of filters change, 1x1 convolutions can be applied so that x with n channels can have a different number of m channels. A 1x1 convolution is just doing a dot product over the channels of a pixel (if 4 input channels, will do a dot product of each corresponding pixel location among the different channels).


13. Visual of 1x1 convolution as a vector dot product ![1x1 Convolution](https://github.com/Nick-palmar/fastai_deep_learning/blob/main/images/1_by_1_conv.png?raw=true)

14. Creating a 1x1 convolution with nn.Conv2d:

```
path = untar_data(URLs.MNIST_SAMPLE)
Path.BASE_PATH = path
# take a 3 to test manually applying kernels as well as using functional pytorch to apply kernels
test_3 = tensor(Image.open(path/'train/3/49020.png'))
test_3_v2 = tensor(Image.open(path/'train/3/24369.png'))
two_threes = torch.stack([test_3, test_3_v2], dim=0)
two_threes.unsqueeze_(dim=1).float()
# create 3 1x1 kernels to go from 1 channel -> 3 channels
k1 = tensor([0.5]).unsqueeze(dim=1)
k2 = tensor([0.2]).unsqueeze(dim=1)
k3 = tensor([0.8]).unsqueeze(dim=1)
one_b_one_kernels = torch.stack([k1, k2, k3], dim=0)
# take a single inpu channel so unsqueeze the 1st axis
one_b_one_kernels.unsqueeze_(axis=1).float()
res_1 = F.conv2d(two_threes.float(), one_b_one_kernels.float())
for outputs in res_1:
  show_image(outputs[0])
  show_image(outputs[1])
  show_image(outputs[2])
```

After experimenting, the shape of the image remains the same after a 1x1 convolution, the only thing that changes is that there are a different number of outputted channels. 

15. The noop function returns the same as the original value (input unchanged). 


16. Figure 14-3 is showing why exactly resnets are more effective than cnn for training purposes. Fig 14-3 shows that the topology of the cnn loss function is very bumpy with many local minimums and a very tough terrain to nagivate in order to reach the global minimum. By contrast, the resnet topology is quite smooth, except for one specific section with an obvious global minimum. This further proves the point that the residual function F(x) = H(x) - x is much easier to optimize than the underlying function H(x) (ie. resnet block versus conv block). 


17. Top 5 accuracy is a better metric than top 1 when images contain multiple categories of labels which may be confused or mislabelled - top 5 reduces the risk of these errors skewing the results. 

18.  The 'stem' of a CNN are the first few layers of a CNN - layers with the most computation occuring but fewest parameters. 

19.  Since the first few layers of a CNN contain the most amount of computation, plain convolutional layers are more effective than resnet blocks as they simplify the beginning  layers by having less computation. The later layers have more parameters and is where skip connections can be advantageous, hence why the stem is plain conv layers while the body of the newtork contains resnet blocks. 

20. A bottleneck block differs from a plain resnet block as it provides layers in the form 1x1 conv -> 3x3 conv -> 1x1 conv (3 conv layers stacked, with the first and last being 1x1 convs). 


21.  A bottleneck block is faster since 1x1 convolutions are faster, so it is reducing the number of channels with the first 1x1 conv layer, then applying a 'regular' conv layer, then exploding the results back up using a fast 1x1 conv layer. With respective to filter size, this is also a faster/more efficient process because more filters can be produced in the same amount of time (due to the 1x1 conv that is diminishing and then computation occuring in the diminished number of channels after the first 1x1 conv layer). 

22. Fully convolutional nets and nets with adaptive pooling in general allow for progressive resizing since they apply an adaptive pool at the end of their process and are thus able to collapse a grid of activations of any size into a single value by taking the average of the remaining activation grid (no matter if it is shaped 10x10 or 2x2). Since the number of output channels will be the same, a final dense layer with this number of output channels can provide the network's output. This allows for different sized inputs to be passed and trained on the model (such as for progressive resizing in training). 