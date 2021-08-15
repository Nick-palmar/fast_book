
## Lesson 4 questions

1. Greyscale images are represented on a computer by pixel values between 0 (white) and 255 (black). Anything in between are shades of grey. These pixel values are held in a matrix (rank 2 tensors). Colour images would be represented by three stacked matricies contianing pixel values for red, green, and blue. These can also take on values from 0 to 255. As such, colour images would be held in 3D tensors (rank 3 tensors). 

2. The MNIST data set has a separate folder for validation and training data sets. This is a common layout for many ML datasets. The training items can be accessed using fast.ai api by calling (path/'train').ls() or (path/'test').ls(). As such, the training and testing set are simple to keep apart for the training process (avoid overfitting). Furthermore, the validation set can report a metric rather than a loss function to be interpreted by humans. Within each test and train folder, there are folders for each image as jpg files. In this specific dataset, there are only subfolders for 3s and 7s. 

3. The "pixel similarity" baseline model of classifying digits works by first getting all the training images, turning them into pytorch tensors, and using the pytorch .mean() function through all the images to find the mean pixel values of each respective training digit. Then, we compare a single image to each of our 'average' models by using the L1 norm (mean average error). Finally, we compare the L1 distances and decide on the digit based on which is closer giving us a tensor of true (is a 3) or false (is a 7). Then, we turn this tensor into a float and take the mean to decide the accuracy of our model compared to validation 3s and 7s.  

4. List comprehensions are a feature in python that allow for a list to be created from a for loop expression within the list. To create one that selects odd numbers from a list and doubles them, I would do:
```
list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

odd_doubled = [(num*2) for num in list if num % 2 != 0]
```

5. A "rank-3 tensor" is a 3 dimensional tensor (rank means the number of dimensions of a tensor). For images, it means that each image has a length and width (2D rank 'matrices') and all the images are stacked on top of eachtoher to create a third dimension (vizualied as a cube).

6. Shape is length of each axis while rank is the number of axes. Rank can be computed from shape by taking the length of the shape. 

7. RMSE and L1 norm are loss functions used to compare the distance betweem predicted and expected results. L1 norm takes the absolute difference between values and computes the mean (MAE) while RMSE takes the mean of the difference squared, and then squareroots them to find the distance. 

8. Calculations can be applied to many numbers at once using NumPy arrays or pytorch tensors which are built for performance at C sepeeds rather than pure python. On the other hand, python loops are very slow. 

9. Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.
```
tns = tensor([i for i in range(9)]).view(3, 3)
tns *= 2
# take only botttom right elements: row 1 to end, columns 1 to end
tns[1:, 1:]
```

10. Broadcasting in pytorch is a trick built in to tensors which will allow for comparisons for tensors of different ranks. That is, the smaller tensor is expanded to match up with the larger tensor during broadcasting. Pytorch will just pretend to expand the smaller tensor by reusing it multiple times; it will not allocate the extra memory. Broadcasting allows for operations between tensors of different ranks (but their shapes should still match in some form to avoid errors). 

11. Metrics are generated using the validation set (not training) because metrics sholud not cause overfitting and are for humans to understand how good the model is doing. In the training loop, the model uses a loss function to determine how close it is to the training data. 

12. SGD is stochastic gradient descent, which is a method for optimizing the parameters of a model. This is done by first initializing parameters, then making predictions using these initially random parameters, finding the loss given training data, taking the gradient (lowest rate of change of the loss function with respect to a certain parameter), and then stepping in the direction of the gradient (steepest descent) so that the parameter gets closer and closer to reaching the optimal point. 

13. SGD uses minibatches in order to be both accurate and efficient when calculating the loss in the training data set. Minibatches are the middle ground between calculating loss on a single item (not informative, possible bad gradients), and the entire data set (very informative and accurate but takes too long). Minibatches are also more efficient on GPUs that can run processess on multiple items at a time. 

14. The seven steps in SGD for machine learning are randomly initializing parameters, 2. making predicitons based on parameters, calculating how good the preds were using a loss, calculating the gradient of the loss function to understand in what direciton the loss chnages the most rapidly wrt each of the parameters (using calc + chain rule), stepping the parameters in the direction of the gradient, repeating back to step 2, and finally stopping when reaching a certain point in the training process. 

15. The weights in a model are randomly initialzied using .requires_grad_() in pytorch in order to be able to take the partial derivative of the loss wrt each parameter (after calculating the gradient). 

16. "Loss" is a function that is used for automated learning that will return lower values when given good predictions and higher values when given bad predictions (compared to the training data). It should be a funciton with a meaningful derivative at all points. In the end, it is a function that should be optimizable using it's gradient (SGD, no flat or infinite slopes in the function). 

17. Using a high learning rate can cause step sizes to be too large and for the SGD algorithm to diverge.   

18. A gradient is a mathematical tool in calculus which is used to define the greatest rate of change of a function with respect to each of it's parameters as a vector. In the ML context, it is a measure of how the loss function changes with respect to changes in the weights of a model. It gives the direction for optimization of a function by altering parameters in direcitons of steepest descent (lowest loss). 

19. No, gradients can be calculated very quickly by pytorch. All that needs to be done is to use .requires_grad_() on the parameters and then call loss.backward() on the loss function to calculate the gradients of any parametes with requires_grad_() applied to them.  

20. Accuracy cannot be used as a loss function because it contains both zero and infinite slopes. As such, the weights of the model will not be able to update according to gradients with zero/inifite parital derivatives. Furthermore, we require the loss function to change with small changes in parameters; by slightly altering the parameters the accuracy may not change, resulting in zero gradients. A good loss must be very sensitive to small, good changes of the parameters. 

21. The sigmoid function takes a range of infinite inputs on the x axis and condenses it into the range of 0 to 1. The current loss function assumes values to be between 0 and 1 so the sigmoid function is useful to transform the values into this range. 

22. A loss is used for the automated learning process for the machine to know how good the predicitons are (calculated in each step of training, must have good derivatives), while metrics are numbers that humans care about and are only shown at the end of each epoch to show how the model is really doing (model performance). 

23. The function to calculate new weights using a learning rate is: new_weight = old_weight - lr*gradient

24. 