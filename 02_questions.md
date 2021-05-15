## Lesson 2 Questions 

1. A bear model might work poorly in production due to structural or stlye differences if the data is coming in through video (as opposed to images), there is night time, the camera has low resolution images, or bears are in rare places (behind other objects) which makes them harder to spot 

2. Text models have a great deficiency in generating correct reponses - however it can generate similar style which can make it seem like it is correct (but this is quite dangerous because it is not accurate)

3. Societal impacts of text generation are speading false news can causing conflicts over false data, when these problems should not be existing 

4. If the mistakes of a model could be harmful, it is good to combine but ML with human intervention to allow the model to do it's just with active corrections from humans in case it is wrong. For example, sending high priority cases for review if an anomaly in a CT scan for strokes is identified. 

5. Deep learning is good for tabular data that contains high variety/natural languarge (book titles or reviews) and also high-cardinality categorical columns (postal codes; where there is not much repetition). 

6. Using an ML model as a recommendation system has the key downside that they only tell you what products a user might like - as oppossed to what products may be helpful for the user. These 'predictions' may be obvious and not very helpful (such as recommmending single books from a set of books you already own or recommending books that you already know about). 

7. Drivetrain approach: Defined objectives (what is the goal) -> levers (what inputs can be controlled, what actions can we take) -> data (what data can be collected) -> models (how the levers influence the objective, the combination of objective, levers, and data for the model)

8. Steps of a drivetrain for a recommendation system: Objective: Increase sales by creating good customer recommdation for products they would not have purchased w/o the recommendation. Levers: The rankings of the recommendations; what will be shown to the user. Data: Must collect new data to drive new sales through random experiments for a wide range of customers (data from the experiments can be passed to the model). Model: 2 models will be built for purchase probabilities - one for seeing an item and one for not seeing it. 

9. Done

10. DataLoaders is a class that takes whatever data you pass to it and passes it along as train and validation data

11. To create data loaders.. tell it the types of the inputs, how to get the inputs, how to label the inputs, and how to split the inputs (create validation set).

12. The splitter param splits the data into a train and validation set based on the validation percent you specify.

13. To ensure random split is always the same, set a seed for the function 

14. independent -> X, dependant -> y 

15. Crop will choose a specific section of the image to a specified size, pad will fill the borders of an image to fit in a certain size, and squish resize reshape the image to fit in a specified size (in turn squishing/streching the images). The best approach may be to use none of these- rather random resize crop which will crop certain parts of an image multiple times so that the same image is seen various times in one epoch (with different croppings); this can help to learn image features. 

16. Data augmentation is the process of taking data and altering it with random variations such that the meaning of the data does not change but it does appear different. For images, this could include rotating, flipping, warping, etc. Useful for models to understand basic concepts - generalizes results

17. item_tfms works on the whole data on the CPU while batch_tfms will perform the transformations on the batch in a tensor on the GPU (so it is much faster)

18. A confusion matrix is a 'grid' that has expected outcomes on the vertical axis and predicted outcomes on the horizontal axis (for categorical problems). This way, it can be visualized how categories were predicted - tells us which classes may be **confusing**. 

19. Export saves both the preprocessing process for input data, as well as a NN (architecture) with the correct parameters (trained)

20. Using model to make predictions - inference

21. IPython widget - JS + python combinaions allowing for interactivity in jupyter notebooks

22. Use CPU for deployment when predictions are made individually and sequentially (most of the time). GPUs can be used in situations where inference is required for many user's data. Downsides include having to wait (since the GPU requires many inputs to be worthwhile), memory management, and batch queqing. 

23. The downside of deploying app to a server instead of client is that you require network connectivity while may cause for increased delays in running a NN. 

24. When rolling out a bear system into practice, problems include... model being too slow for predictions to be useful, night time images, low resolution images. 

25. Out of domain data is data that is very different in production from what the model saw in training

26. Domain shift is when the type of data a model sees changes over time - change in the types of customers a company attracts

27. The 3 steps in the deployment process are... 
    1. Manual Process: Where the model is running beside normal operations and is not being used to drive any decisions - humans should be validating the model to ensure that the results obtained make sense
    2. Limited Scope Deployment: Deploy the model at specific times, in specific settings. The scope of model usage is controlled and all results are still checked by humans before drawing any conclusions from the model,
    3. Gradual Expansion: Slowly roll out to different scopes, having a robust method of reporting the results from the model to be aware of significant changes between model and manual processes (large changes should raise red flags). You should also constantly consider what can go wrong and ensure that any findings are well documented.
    

## Further Research 
2. When data augmentations change the labels for the data, it may be best to avoid it. For exmaple, flipping the number '9' can change it into a '6'! In medical applications, similar cases can occur so caution must be taken.