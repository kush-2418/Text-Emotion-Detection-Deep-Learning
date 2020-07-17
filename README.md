# Text Emotion Detection using Deep Learning

1. [ Demo ](#demo)
2. [ Overview ](#overview)
3. [ Dataset ](#data)
3. [ Installation](#install)
4. [ Run ](#run)


<a name="demo"></a>
### Demo
#### Link https://text-emotion-detection-dl.herokuapp.com

<a name="overview"></a>
### Overview

This Text Emotion Detection project detects the emotions from the given text and classifies them into set of 6 emotion classes i.e. Anger, Fear, Joy, Love, Sadness and Surprise. The model has been created using deep learning based LSTM architecuture and other one using CNN architecture. The words are embedded into vector form using pre-trained Glove word vectors. The LSTM model achieves the accuracy of 93% and loss 0.11 whereas the CNN model achieves the accuracy of 90% and loss of 0.21. The model has been deployed using Flask framework over the Heroku server.

<a name="data"></a>

### Dataset
Download dataset using the given code

```python

!wget https://www.dropbox.com/s/607ptdakxuh5i4s/merged_training.pkl
dataset = pickle.load(open(merged_training.pkl,"rb"))
```
### Glove Embedding
Download the pre-trained Glove Word Embedding using the given code
```python
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip
```
<a name="install"></a>
### Installation

The Code is written in Python 3.7. To install the required packages and libraries, run this command in the project directory after cloning the repository:

> pip install -r requirements.txt

<a name="run" > </a>
### Run

Create an environment and clone this repository. To run this project run a command into terminal :

> python app.py


```
