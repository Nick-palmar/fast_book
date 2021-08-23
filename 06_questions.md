## Lesson 6: Other CV problems

1. Multi-label classification could improve the usability of the bear classifier for the classifier to be able to predict 'no bear' if the image it was shown does not include a bear. This is due to the fact that multi label classificaiton does not require probabilites that add up to 1 for the predicitons, so using multi label classification we can set a cutoff that is high enough for the model to say 'no bear' if it does not have high enough bear probabilities. Furthermore, if multiple types of bears appear in an image, multilabel classification can deal with this.

2. In multi-label classification problems, the dependant vairable is often one-hot encoded such that labels present have a value of 1 at the respective indices and all other labels have a value of 0 at the other indices. This allows for the output tensors to have the same size independant of the number of categories present. 

3. How do you access the rows and columns of a DataFrame as if it was a matrix?
```
# access rows of DataFrame
row_1 = df.iloc[row_number]

# access as matrix like m r,c:
matrix_index = df.iloc[row_num, col_num]

# access columns of df 
col_1 = df[col_name]
```

4. To get column by name, refer to last line of code in question above. 

5. A dataset is a collection that returns a tuple for each item in the form (independant, dependant) and a dataloader is an iterator which creates minibatches as a tuple of independant varibales and a tuple of the dependant variables. A data set is usually created before a data loader and builds up into a data loader for use with the model - it is an extension of dataset functionality. 

6. A DatasetS object normally contains a training and validaiton dataset 

7. A DataloaderS object normally contains a training and validaiton dataloader

8. Lambda in python is an anonymous function; it allows us to create single line functions without a name or explicit return statement. However, lambdas are not serializable so they should not be used in deployment and production. 

9. Inside the data block, the get_x parameter can be passed a function which customizes how to get the independant variable, and the get_y parameter can be passed a function that customizes how to get the dependant variable. 

10. Softmax is not an appropriate output activation function when using one hot encoded targets because softmax attempts to choose one specific target due to the exponential nature of softmax. Also, softmax sums probabilites to 1 which may not be ideal when multiple objects appear in the image (since all the probabilites will be high). Conversely, when no objects appear in the image we want lower probabilities which should be lower than 1. 

11. nll_loss is not appropriate when using one hot ended targets because it will only return a single label for an item, not multiple labels. 

12. nn.BCELoss differs from nn.BCEWithLogitsLoss in that nn.BCELoss does not apply sigmoid to the inputs while nn.BCEWithLogitsLoss does apply sigmoid before applying cross entropy. 

13. Regular accuracy cannot be used for a multi-label problem because it predicts a single label using argmax which is not useful when there are multiple labels. The fix for this is to use a threshold to predict a class so that multiple classes can be predicted if the prediction is above the set threshold value. 

14. It is ok to tune a hyperparameter on the validation set if the relationship with the hyperparameter smooth so the value can be easily picked without making any strange assumptions. 

15. y_range is implemented in fast.ai using a modified version of sigmoid to smooth the values between the low and high values provided. The implementation is as follows:
```
def y_range(inputs, low, high):
    sig_inputs = torch.sigmoid(inputs)
    return sig_inputs*(high-low) + low
```

16. A regression problem has a continuous variable(s) as the target. Usually, this means a float value or a list of float values. For these problems, the loss function is normally mean sqaured error and this can also be the metric (easier to square root the result for the metric - MSE for loss and RMSE for metric). 

17. For fast.ai to know that it should move the target point after data augmentation, we should set the second block of the DataBlock as PointBlock. This way, augmented images have target points moved correspondingly; fast ai is one of the only libraries to allow for this. 

