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

8. 