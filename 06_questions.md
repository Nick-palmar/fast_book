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

9. 