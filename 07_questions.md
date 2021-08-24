## Lesson 7: Training State-of-the-Art Models

1. ImageNet is an image classification dataset which contains 1.3 million images of various sizes with 1000 categories, while Imagenette was created by fastai and is a subset of ImageNet with only 10 very different categories. The reason that this is useful is that algorithmic tweaks on imagenette generalized well to imagenet. It is better to experiment on ImageNette since it is a smaller data set which can allow for quick experimentation and iteration of prototypes. When going for the final model, working on ImageNet would be better since there is more data here for the model to learn from and generalize to - however the model will take much longer to train. 

2. Normalization is converting data so that it has a mean of 0 and a standard deviation of 1. 

3. We previously didn't care about normalization when using a pretrained model because cnn_learner in fastai is capable of figuring out the proper normalization statistics for a dataset and apply those normalization statistics to the dataset in order for new training data to align with the weights from previous, normalized training of the data.

4. Progressive resizing is the process of training a model starting with small image sizes, and progressively increasing to larger image sizes in a transfer learning-ish manner. What this means is that we start training the model with small images to pick up basic features, then as we increase image size, we fine tune (freeze, fit, unfreeze, fit) treating the increase in size as if we are doing transfer learning to generalize to better results. The process should not make images larger than their original sizes. The outcome is faster training for the initial, smaller images are more accurate results as we train the larger images.

5. **To do in notebook to test**

6. Test Time Augmentation is creating multiple version of an image in validation using data augmentation and then taking the average or maximum of the predictions. This way, the validation set does not only include one variety of augmentations (centre crops in the case of fastai when doing RandomResizeCrop). In fastai, you use TTA by doing: 
```
preds_TTA, targs_TTA = learn.tta()
accuracy(preds_TTA, targs_TTA)
```

7. TTA at inference is slower than regular inference because TTA is augmenting the validation set to create multiple version of a single image - as such the model must make predicitons on more images so TTA will take longer. 

8. Mixup is a data augmentation technique that can lead to higher accuracy especially when training a model from scratch and training for a larger number of epochs. For each image in a dataset, mixup randomly chooses another image in the dataset, then picks a random weight and multiplies the initial image by the random weight and adds (1- the random weight) times the other image. As such, this forms a linear combination of the two source images for the new image. The same linear combination is applied to the target, so the target values get pushed as probablities between 0 and 1 (for example, 0.3 of one image and 0.7 of another become the targets). 

9. Mixup prevents the model from being too confient because as a result of combining images, the target values do not take on such high values (no longer 1 because both probabilites from mixup add up to 1). Thus, the model is trained to predict lower confidence scores and will not become overly confident when making predictions in the future. Furthermore, since mixup is different in each epoch, the model will not overfit on the training data as the 'new' images from mixup in each epoch will cause the model to be less confident but generalize better wehn training from scratch. 

10. Training with mixup for 5 epochs ends up being worse than training without mixup because mixup causes the model to try to predict 2 labels rather than 1, and must predict the weight for each one. Due to the nature of this more difficult task, it requires more epochs of training for mixup to generalize well to a dataset. If trained for fewer epochs, it will not be as accurate. 

11. The idea behind label smoothing is to encourage the model to purposefully be less accurate in order to account for mislabelled data and generalize better during inference. The model can be elss accurate by slightly increasing 0 probability targets and slightly decreasing 1 probability targets. Without label smoothing, models tend to become overly confident in training so they predict things overly confidently in inference even if the data is something the model has never seen before. Label smoothing addresses this problem by reduing the target accuracy for the model to not predict so confidently. 

12. Label smoothing can help account for mislabeled data or bias in the data. In practice, data will never be perfect due to simple labelling errors or bias in labelling that make it difficult to provide a single, objective answer for a label. Label smoothing addresses this data issue by decreasing model confidence such that the model does not become overl confident on training data. If there is bad data, the model will not become overly confident on this data so it will generalize better. 

13. When using label smoothing with 5 categories, the target associated with index 1 can vary depending on the data error predicted (*epsilon*) and on whether or not index 1 has the correct label. If index 1 has the correct label, then it's value after label smoothing should be 1 - *epsilon* + (*epsilon*/**N**), where *epsilon* is a value between 0 and 1 according to predicted error in data. If it is not the correct label, index 1 after label smoothing should contain *epsilon*/**N**. 

14. The first step to take when prototyping a quick experiement on a new dataset is to pick a subset of the data that is representative of the entire dataset and experiment on it (ideally so that results from the inital prototypes will generalize well to the entire dataset as fastai was able to do when going from Imagenette -> Imagenet). 