## Lesson 5 Questions: Pet breed classification

1. The process of first resizing on the cpu and then augmenting by performing 'warping' operations (rotating, streching, etc) is known as presizing in fast.ai. The problem with normal data augmentation is that empty edge zones are added or data can be made worse (from various different transforms occuring at different times). By first resizing to a large size on the CPU and then selecting random crops (with streching) on the GPU, we are allowing for no empty edge zones with fewer operations on the data. This way, training images are kept high quality and able to train quickly. 

2. **TO DO LATER**

3. In most deep learning datasets, data is commonly provided in one of these two ways: individual images which can be in specific folders and have specific file names to give information about the data contained in the file (what the target variable is), or a table of data that provides the connections between data in the table and data in other file formats (such as text documents or images). 

4. 