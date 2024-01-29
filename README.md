# img2prompt project

In this research project we used a Recurrent Neural Network (RNN) and a Convolutional Neural Network (CNN) utilizing stable diffusion to deduce prompts used to generate images. The goal is to predict what prompt or series of prompts were used to generate an image using the Stable Diffusion deep learning, text-to-image model. The dataset was taken from a much bigger batch of AI generated images with prompts, namely DiffusionDB. The source used was Hugging Face, which contains a big dataset split into smaller files. We opted for one of those files of around 10GBs of data. It was chosen since it was big enough to train a model, and small enough to fit on a regular personal computer. The dataset is filled with 10,000 AI generated images along with various types of prompts used to generate them. We went for a hybrid approach, combining the CNN’s image analyzation capabilities with the RNN’s text generation capabilities. 

## Initialization (initialization.ipynb)
The dataset that we decided to use came from a larger dataset containing two million images and
prompts. Unfortunately, due to memory limitations we were not able to use all of the data.
Instead, we settled on the first 10,000 rows of the dataset. This dataset also included various
metadata columns, but these were dropped as we were interested solely in the images with their
corresponding prompts.

## Pre-processing (preprocessing.ipynb)
Before developing a model to train the data on, it is important to preprocess the data to ensure
that it is suitable for the model to train on. Since we are working with both images and text, we
had to take different steps to preprocess each of them.

### Images
For the images, one notable detail was that the sizes were different for each image. For
consistency among the images, we decided to resize every image to 128x128 pixels. This also
significantly decreased the memory needed to store the dataset. Next, it was necessary to convert
the images into a numerical representation, since Convolutional Neural Networks require tensors
as an input. Since all images are in RGB color, meaning they have three color channels, the
tensor size for every image ends up being 128x128x3 where every value depicts one pixel with a
value from 0 to 255 which represents the intensity of that pixel. Furthermore, these values are
normalized, meaning that they are rescaled to a range of 0 to 1 instead. Since neural networks
rely on calculating gradients, normalizing the values can make the training process more
efficient. At this point, the images are ready for training.

### Text
Next is the text data. The first issue was that a lot of prompts contained unwanted symbols such
as question marks, exclamation marks, commas, and other symbols. These were removed so that
only the words remain. To prepare the text data for training, we decided to use word embeddings;
a word embedding is nothing more than a numerical representation of a word. This was achieved
using an algorithm called Word2Vec, which trains on an input of text, and learns associations
between words as well as their semantics. Since text does not require much memory
space, this Word2Vec model was trained on all two million prompts from the original dataset to
ensure a larger vocabulary size. After this process, every word of every prompt is converted to a
vector containing values that represent the meaning of every word. These values range from
around -7 to +7. However, these will not be normalized because the values are significant and we
can not lose them.

## Training (training.ipynb)
Before developing our model, we need to split our data into training, validation, and testing sets.
This will aid with evaluating the model’s performance as well as preventing overfitting. We
settled on 80% of the data going to training, 10% validation, and 10% testing.

To create our model, we combined a CNN with an RNN. The architecture of this model consists
of three convolutional layers with max-pooling layers after them to highlight important features,
followed by a dropout layer to help prevent overfitting by disabling random nodes. This is the
basic architecture of our CNN, which outputs feature maps between the images and prompts.
This output is then flattened and used as an input to the RNN which consists of a long short-term
memory (LSTM) layer and two dense layers with 100 nodes each that act as hidden layers. Since
an LSTM can process sequences of data, it is ideal for text. See appendix A for the code.
Finally, the output is processed through a custom activation function based on the tanh activation
function. A tanh activation function was used to ensure that the output is in range [-1, 1].

However, since our aforementioned vectors of text data contained values from between
approximately -7 and +7, and we were unable to normalize them to -1 and 1, we need to
customize this activation function. Essentially, the goal is to output in a range from the minimum
value to the maximum value in all word embeddings. Mathematically, the new activation
function can be represented as seen below:

$𝑓(𝑥) = 𝑡𝑎𝑛ℎ(𝑥) · \frac{𝑚𝑎𝑥(𝑤𝑜𝑟𝑑E𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔𝑠)−𝑚𝑖𝑛(𝑤𝑜𝑟𝑑E𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔𝑠)}{2} + \frac{𝑚𝑎𝑥(𝑤𝑜𝑟dE𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔𝑠)+𝑚𝑖𝑛(𝑤𝑜𝑟𝑑E𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔𝑠)}{2}$

This new function ensures that the values our model outputs are in the same range as the word
embeddings’ minimum and maximum values, which allows us to use the same Word2Vec model
to find the closest words for each output. This way, we made the model output about 10 relevant
words related to the input image instead of full sentences, since we were unable to make it output
coherent sentences.
Moreover, the loss function that is used to evaluate the model is the cosine similarity. The cosine
similarity is a measure of similarity between two vectors: 1) the generated words, and 2) the
words the model was trained on. It is represented by the angle between two vectors, which
describes how close they are to each other.

### Limitations
As it can be seen from the graphs in `training.ipynb`, the model did not perform very well. The first
problem that might be the reason for such a poor performance is a small training dataset. The
archive where we got our dataset from encompasses over 6TB of data (images + their prompts).
We used only 10,000 images, which was roughly 10GB worth of data, before shrinking it. To achieve a higher degree of generalization with the model, it would be preferred to use even more data.
Furthermore, after having a look at the pictures available to us, i.e the 10GB dataset, we noticed
that there is considerable class imbalance in the dataset. Even though the entire archive has representative images for all sorts of categories,
our set of pictures had plenty of examples in some categories but no examples in some others
(e.g. we encountered multiple variations of Walter White interacting with some object or person).
Because some subjects were overrepresented and some other subjects were not present at all, the
model was likely not able to train and gain information uniformly.
