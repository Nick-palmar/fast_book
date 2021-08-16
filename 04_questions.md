
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

24. A dataloader is a class from fast.ai that can take a collection (in python like a list, tuple, etc) and make it an iterator over many batches. In the context of ML, it is generally used with a list of tuples being converted into an iterator so that the training loop can iterate over an object of class dataloader in specific batches. 

25. Basic steps in SGD
```
for batch_x, batch_y in data_loader:
    # apply the model to make predictions
    preds = model(batch_x)

    # calculate the loss based on the predicitons. Use sigmoid here
    loss = mnist_loss(preds, batch_y)

    # calculate gradients with back propagation
    loss.backward()

    # step each parameter and clear part of the gradient after applying
    for param in params:
        param.data -= param.grad * lr
        param.grad = None
```

26. Create a function that, if passed two arguments [1,2,3,4] and 'abcd', returns [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]. 
```
def create_iterable_tuples(col_1, col_2):
  it_tup = list(zip(col_1, col_2))
  return it_tup
```

This output structure is special for ML since you can have one item in the tuple be the training data, while the other item is the expected result for that training data. This allows for training to be simple (data paired with target). 

27. View in pytorch changes the shape of a tensor without changing its contents. In the example, a rank-3 tensor was changed to a rank 2 tensor using view. 

28. Bias parameters in a NN allows for flexibility in the model. If the input is zero, the equation mx+b will always be zero (since mx = 0). By adding b, the bias term, this can be changed. 

29. The @ operator in python is used to perform matrix multiplicaiton. 

30. The backward method in pytorch is used for backpropagation to calculate the gradient of parameters from before that have requires_grad_() by making use for the chain rule along with parital derivatives from multivariable calculus. 

31. In pytorch, the gradients will accumulate if they are not zeroed out which will lead to incorrect reuslts - thus they must be set to zero after each step. 

32. Learner must be passed a DataLoaders, which can be created through the DataLoader class with a training and validation data loader, a model, an optimization function, a loss function, and can optionally be passed a metric. 

33. Show Python or pseudocode for the basic steps of a training loop.
```
def train_model(model, epochs, train_dl, valid_dl):

    # loop the number of epochs
    for i in range(epochs):
        # train an epoch by looping through a batch
        for train_x, train_y in train_dl:
            preds = model(train_x)
            loss = calc_loss(preds, train_y)
            loss.backward

            for param in params:
                param.data -= param.grad*lr
                param.grad = None
        
        # get the epoch Accuracy, assume batch_accuracy() function calculates the accuracy of any batch
        acc_list = [batch_accuracy(valid_x, valid_y) for valid_x, valid_y in valid_dl]
        acc = torch.stack(acc_list).mean() 
        print("Epoch accuracy:", acc)
```


34. 'ReLU' is a rectified linear unit and is used as an activation function in ML (a non-linear layer between two linear layers). It takes all negative values and assigns a value of 0 and take all positive values and keeps them at their current value. Between -2 and 2, we have the piecewise function y=0 for -2<=x<=0 and then y=x for x>0. 

35. An activaiton funciton is a non-linear function used to allow for non-linear relationships to be estimated very precisely by a nueral network. 

36. F.relu is a python function while nn.relu is a module (object that inherits from a pytorch class). They can be used in the same way.  

37. Although the universal approx theorem shows that any function can be approx as closely as needed using a single nonlinearity, we normally use more bc it is proven by practitioners that by using more layers, we can use smaller matrices and get better results than we would with larger matrices and fewer layers. It is a perforamnce trade off due to how this all works in pracitce rather than a theoretical limitation (less memroy, faster training, better performance). 