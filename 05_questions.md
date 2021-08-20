## Lesson 5 Questions: Pet breed classification

1. The process of first resizing on the cpu and then augmenting by performing 'warping' operations (rotating, streching, etc) is known as presizing in fast.ai. The problem with normal data augmentation is that empty edge zones are added or data can be made worse (from various different transforms occuring at different times). By first resizing to a large size on the CPU and then selecting random crops (with streching) on the GPU, we are allowing for no empty edge zones with fewer operations on the data. This way, training images are kept high quality and able to train quickly. 

2. **TO DO LATER**

3. In most deep learning datasets, data is commonly provided in one of these two ways: individual images which can be in specific folders and have specific file names to give information about the data contained in the file (what the target variable is), or a table of data that provides the connections between data in the table and data in other file formats (such as text documents or images). 

4. New methods for fast ai L class:
```
l = L()

l.append(5)
l.append(4)
l.append(8)

l2 = L(range(50))

l.append(5)
l.append(5)

unique_l = l.unique()
unique_l

l_filter = l2.filter(lambda num: num < 10)
l_filter

l_mapped = l_filter.map(lambda num: num*2)
l_mapped

dependant = [1, 2, 3, 4]
target = ['a', 'b', 'c', 'd']

d_loader = L(dependant, target).zip()
d_loader
```

5. Using the pathlib module in python  
```
# path = path.relative_to('/root/.fastai/data/oxford-iiit-pet')
path_list = [subdir for subdir in path.iterdir() if subdir.is_dir()]
path_list = L(path_list)
path_roots = path_list.map(lambda path: path.root)
print(path_list)

path_data = path_list[2]
mnist_small = path_data/'mnist_train_small.csv'
mnist_small.exists()
```

6. Image transforms can degrade the quality of the data by introducing empty sections in the image (useless for learning) or applying interpolations as part of the transformation that are not part of the image. These interpolations are of lower quality and thus degrade image quality. 

7. After a DataLoader is created, fast.ai provides the d_loader.show_batch() method in order to see images in a data loader. If the parameter unique is set to True, it will show the augmentations done for a certain image in the data loader (training set).

8. fast.ai provides the dataoader.summary(path) method to show a summary of the data loader for a specific path. 

9. No, models should be trained as soon as possible. This is so that a good baseline can be taken to see if a simple model is enough for your needs, and also to check if the data is not training the model (possibly due to dirty data). We can then look to see where the model is going wrong and fix these problems as part of the cleaning (as shown in lesson 2). 

10. The two parts that are combined in cross entropy loss in pytorch are log_softmax() which first applies the softmax function (e to the power of the current prediction divided by the sum of e to the power of each prediction) and then takes the log of these values. This gives numbers between negaitve infinity and zero (from below). The second part is nll_loss, which indexes the correct category and takes the negative such that the loss is now a single value between 0 (from above) and positive infinity. 

**Below is some sample code testing the loss function concepts**
```
# create a tensor with 3 categories and standard deviation 3
test_tens = torch.randn(12)*3
sample_acts = test_tens.view(-1, 3)
sample_acts

# calculate the softmax of the tensor
softmax_acts = sample_acts.softmax(dim=1)

# take the log of the tensor
log_softmax_acts = softmax_acts.log()

# index using nll_loss
targets = tensor([0, 1, 2, 1])

cross_entropy_loss = -log_softmax_acts[range(sample_acts.shape[0]), targets]
cross_entropy_loss, cross_entropy_loss.mean()

# use cross entropy loss with pytorch
nn.CrossEntropyLoss(reduction='none')(sample_acts, targets), nn.CrossEntropyLoss()(sample_acts, targets)
```

11. Two important properties that softmax ensure is that they all add up to 1 (the probailities add up to 100%) and one specific category is favoured over the rest due to the nature of exponential functions. 

12. You might not want your activation to want these two properties if the data for interence is not part of any of the categories (no label should activate) in training or if you have multiple labels as part of the classification. 

13. Calculate the exp and softmax columns of Figure 5-3 yourself (i.e., in a spreadsheet, with a calculator, or in a notebook).
```
outputs = tensor([0.02, -2.49, 1.25])

exp = torch.exp(outputs)
softmax = exp/exp.sum()
softmax
```

14. Can can't use torch.where for loss func with >2 labels because torch.where can only be used to select between to labels in an if-else manner. With more than 2 categories, we must choose a specific category out of 3+ possible options which cannot be done with the .where function. 

15. log(-2) is not defined because the domain of log is real positive numbers (not including zero). 

16. 

