## Lesson 1 Questions 

1. 
    - F; deep learning does *use* lots of math (lots of addition and multiplication), but **you do not need** lots of math
    - F; some of the best deep learning models use very little data
    - F; use free gpu on cloud (google collab or paperspace)
    - F; Musk says high school is enough for deep learning

2. Deep learning is the best in medical image recognition, computer vision (autonomous vehicles), NLP (speech recognition), playing games (beats the world champs), recommendation systems

3. The **perceptron** was the name of the first device that used concepts of artificial neurons

4. The requirements for parallel distributed processing (PDP; framework closer to how our brain works) are
    - Set of processing units
    - State of activation; state for the units 
    - Output function for each unit; an output 
    - Pattern of connectivity among units 
    - Propagation rule (transfer patterns throughout units)
    - Activation rule which combines unit's state with inputs to produce an output
    - Learning rule where patterns of connectivity can learn from experience (data, **what else can make up experience?**)
    - Environment for system to run

5. 2 theoretical misundertandings in the field of neural networks were... 
    - single layers were too basic to be useful (implying that NN themselves were not useful)
    - adding a second layer would be sufficient to fix all problems (but these netwoeks were too big + slow to be useful)

6. GPU - graphics processing unit, good for making computations needed for deep learning (better than CPU). Can handle many tasks at the same time. 

7. Cells codes run python and have output below the cell (show work as you progress)

8. Completed

9. Completed

10. Traditional computers have trouble recognizing images in a photo because we would need to spell out every detail for the computer to understand - and we ourselves don't necessarily understand all the details!

11. Weight assignments are choices of values for variables that will define how the program operates from getting inputs to producing outputs. 

12. Weights -> parameters 

13. Done in One Note

14. There are many layers of neurons with different weights which may be difficult to understand, however, there has been lots of research in extracting insight from models and understanding what every layer of a deep NN is doing (CNN layer explanation)

15. Univerisal approximation theorem - a NN can solve any type of mathematical problem to any level of accuary

16. To train a model, you need inputs for the data, labels for the input data, an architecture which is the actual function that will be trained, a loss function to figure out how close the results were to the actual results and an optimizer to reduce the loss function on each iteration by updating the parameters of the function  

17. If the data going into a predictive police model is bias towards a certain ethnicity, the model will learn the bias in the data and will end up making bias predictions. In turn, there bias predictions will be fed back into the model. This is called a positive feedback loop

18. No, we do not **need** to use 224x224 pixel images, these are simply the standard. Inc pixels = inc accuracy = inc training time (accuracy v time/resources trade-off)

19. Classificaiton is when inputs are labelled into certain categories as predictions while regression is when a continuous numerical value is predicted as a result of the inputs

20. A validation set is a set of data used to validate a model after it has been trained on the training set. A testing set is only meant to be used right at the end to validate that the final model works well (has not overfit on validation set as well). A validation set is meant to test the training set (avoid overfitting/bias in training by the NN), while a testing test is meant to test the accuracy of the final model (avoid overfitting/bias in the training by a human on the data). 

21. By default, fastai uses 20% of data as validation set  

22. No, random smaples cannot be used all the time for validation sets. For time series data, random sample would be too simple to predict and would not be indicative of results (since there would be data before and after the results). Instead, the most recent 'time' in the time series data should be held out as the validation set to ensure the model is making predictions on the future - like it will be doing after it is built. **Good validation set should repr. the type of data to be seen in the future.**

23. Overfitting is when the architecture starts to memorize the training data instead of learning the general rules. This means that it will perform very very well on the data it is trained on but will not generalize well on new data. This is a huge problem in ML and there are various techniques to deal with this. It is also the reason that there is a validation set + testing set held out from the training set.  


24. A metric is a means to quantifying the models performance on the testing set using values that humans can easily interpret. It differs from the loss function as the loss function is meant to tell the optimizer how to make better weights (so it must be good for the learning process), but it may not be easy to interpret for humans. Loss -> for training the architecture. Metric -> for human understanding of trained model. Note: They can still be the same in some cases

25. Pretrained models can help by providing an architecute that is already very capable of performing common tasks (don't re-invent the wheel). It allows us to train models with more accuracy, more quickly, less data, less time, and less money 

26. The head of a model (when using a pretrained model) is the part of the model that is removed from transfer learning to perform the specific task. These are new layers tailored to the specific task (when using a CNN, the head are the dense layers for classifying the images while the body retains the convolutional layers from transfer learning)

27. Early layers of CNN find edges, diagonals, gradients - simple patters. There patters are then built up into slightly more complex shapes such as circles and rectangles and the final layers contain complex shapes which can be identified (outlies of animals, car wheels, flower pedals)

28. Image models can be used to understand many types of data, not just photots. For example, sound can be converted into a spetrogram (image) and can then be analyzed by an image model. Fraud detection models can analyze mouse movement (of a user's mouse) and then use these pictures to identify fraud. Images can even be generated from time series data. 

29. Architecture is the **template** for the mathematical function we are trying to fit

30. Segmentation - separating different components in an image; classifying every single pixel in the image. 

31. y_range is used when specifying the range on predictions on a regression model. We need it when we have limits/boudaries on regression problems.

32. Hyperparameters are parameters for the parameters, usually referring to choices like network architecture, learning rates, or data augmentation strategies. **High level choices that affect meaning of weight parameters**. 

33. When using AI for an organization, failure can be avoided by... 
    - creating simple baseline models to test their performance against those of expert vendors  
    - Holding out some of the data from the vendor as a testing set and ensuring that this testing set is reaching the value specified by the company (**not the vendor**)

## Extended Learning Questions

1. GPUs are better than CPUs for deep learning because the can perform many of the same tasks in parallel. CPUs can only do a few tasks (depending on the number of cores) sequentially.