
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

8. 