# Deep Learning Challenge: German traffic signs detector app

## Intro:

This is repository competes for the Kiwi deep learning challenge. Its purpose is to present a command line application that is able to:

- Download images from the *German Traffic Signs Dataset*, organize them into training and test sets and rename the images according to the label of each image.

- Train two different models using this data: A logistic regression model using scikit-learn and a softmax regression model using tensorflow.

- Save the generated models and evaluate their performance (accuracy) on a test set.

- Use the models to classify new images, displaying each image and their corresponding tag.

## Commands:

The main script of this application is **app.py**. To launch this application, please open a terminal and go to the folder containing this file.

The following commands can be called from the terminal:

#### To download and organize the images: 

`python app.py --download`

#### To train a model:

`python app.py --train -m [model] -d [directory]`

Where *model* can either be "model1" (logistic regression model) or "model2" (softmax regression model); and *directory* is the location of a folder with the train set of images. 

**Important:** According to the directories structure proposed for the challenge, there are a *train* folder and a *test* folder at the same level. When the **train** command is excecuted in the app, the program will expect the *directory* argument to be the *train* folder, and will look for the *test* folder automatically.

When excecuted, this command will generate and save a model in the *models* folder, and print in the terminal the accuracy obtained on the training set.

#### To test a model:

`python app.py --test -m [model] -d [directory]`

Where *model* can either be "model1" (logistic regression model) or "model2" (softmax regression model); and *directory* is the location of a folder with the test set of images.

When excecuted, this command will look for the corresponding model (which must be created and saved previously), apply the model to the test data, and print in the terminal the accuracy obtained on the test set.

#### To classify new images:

`python app.py --infer -m [model] -d [directory]`

Where *model* can either be "model1" (logistic regression model) or "model2" (softmax regression model); and *directory* is the location of a folder with the images that need to be classified.

When excecuted, this command will display each of the images contained in the directory (one at the time), with their corresponding predicted label (number and name of the traffic sing).

**Important:** The images must be *.ppm* files.

## What else is in this folder:

This repository follows the challenge's proposed directories structure, so it includes:

- A `reports/` folder, that includes two jupyter notebooks, each one including the process of development and training of one of the models.

- A `models/` folder, that includes a folder for each model. Each folder contains the saved models, that are generating in the training process. Each time the training is made, these files will override.

- The `images/` folder is not included. It should be generated using the `--download` command.


Additionally, two files are included:

- `support_functions.py`: Contains functions and code that are used by the main `app.py` file.

- `requirements.txt`: Contains the python libraries neccessary to run the app.

