# deep-learning-challenge

## Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup. From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as: 

EIN and NAME—Identification columns <br>
APPLICATION_TYPE—Alphabet Soup application type <br>
AFFILIATION—Affiliated sector of industry <br>
CLASSIFICATION—Government organization classification <br>
USE_CASE—Use case for funding <br>
ORGANIZATION—Organization type <br>
STATUS—Active status <br>
INCOME_AMT—Income classification <br>
SPECIAL_CONSIDERATIONS—Special considerations for application <br>
ASK_AMT—Funding amount requested <br>
IS_SUCCESSFUL—Was the money used effectively

## Before You Begin
Create a new repository for this project called deep-learning-challenge. Do not add this Challenge to an existing repository. <br>
Clone the new repository to your computer. <br>
Inside your local git repository, create a directory for the Deep Learning Challenge. Push the above changes to GitHub. <br>
## Files
Download the following files to help you get started: <br>
Module 21 Challenge files <br>
Links to an external site. 
## Instructions 
### Step 1: Preprocess the Data <br>
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model. <br>
Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps. <br>
Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset: <br>
What variable(s) are the target(s) for your model? <br>
What variable(s) are the feature(s) for your model? <br>
Drop the EIN and NAME columns. <br>
Determine the number of unique values for each column. <br>
For columns that have more than 10 unique values, determine the number of data points for each unique value. <br>
Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, Other, and then check if the replacement was successful. <br>
Use pd.get_dummies() to encode categorical variables. <br>
Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets. <br>
Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function. <br>
### Step 2: Compile, Train, and Evaluate the Model 
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy. <br>
Continue using the file in Google Colab in which you performed the preprocessing steps from <br>
Step 1. <br>
Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras. <br>
Create the first hidden layer and choose an appropriate activation function. <br>
If necessary, add a second hidden layer with an appropriate activation function. <br>
Create an output layer with an appropriate activation function. <br>
Check the structure of the model. <br>
Compile and train the model. <br>
Create a callback that saves the model's weights every five epochs. <br>
Evaluate the model using the test data to determine the loss and accuracy. <br>
Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5. <br>
### Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%. <br>
Use any or all of the following methods to optimize your model: <br>
Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as: <br>
Dropping more or fewer columns. <br>
Creating more bins for rare occurrences in columns. <br>
Increasing or decreasing the number of values for each bin. <br>
Add more neurons to a hidden layer. <br>
Add more hidden layers. <br>
Use different activation functions for the hidden layers. <br>
Add or reduce the number of epochs to the training regimen. 

Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance. 

Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.<br> 
Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.<br> 
Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model. <br>
Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy. <br>
Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5. <br>
### Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup. The report should contain the following: <br>
Overview of the analysis: Explain the purpose of this analysis. <br>
Results: Using bulleted lists and images to support your answers, address the following questions: Data Preprocessing <br>
What variable(s) are the target(s) for your model? What variable(s) are the features for your model? <br>
What variable(s) should be removed from the input data because they are neither targets nor features? <br>
Compiling, Training, and Evaluating the Model <br>
How many neurons, layers, and activation functions did you select for your neural network model, and why? <br>
Were you able to achieve the target model performance? <br>
What steps did you take in your attempts to increase model performance? Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation. 
### Step 5: Copy Files Into Your Repository
Now that you're finished with your analysis in Google Colab, you need to get your files into your repository for final submission. <br>
Download your Colab notebooks to your computer. <br>
Move them into your Deep Learning Challenge directory in your local repository. Push the added files to GitHub. 
## Requirements
#### Preprocess the Data (30 points) 
Create a dataframe containing the charity_data.csv data , and identify the target and feature variables in the dataset (2 points) <br>
Drop the EIN and NAME columns (2 points) <br>
Determine the number of unique values in each column (2 points) <br>
For columns with more than 10 unique values, determine the number of data points for each unique value (4 points) <br>
Create a new value called Other that contains rare categorical variables (5 points)<br> 
Create a feature array, X, and a target array, y by using the preprocessed data (5 points) <br>
Split the preprocessed data into training and testing datasets (5 points) <br>
Scale the data by using a StandardScaler that has been fitted to the training data (5 points)
#### Compile, Train and Evaluate the Model (20 points) 
Create a neural network model with a defined number of input features and nodes for each layer (4 points) <br>
Create hidden layers and an output layer with appropriate activation functions (4 points) <br>
Check the structure of the model (2 points) <br>
Compile and train the model (4 points) <br>
Evaluate the model using the test data to determine the loss and accuracy (4 points) 
Export your results to an HDF5 file named AlphabetSoupCharity.h5 (2 points) 
#### Optimize the Model (20 points) 
Repeat the preprocessing steps in a new Jupyter notebook (4 points) <br>
Create a new neural network model, implementing at least 3 model optimization methods (15 points) <br>
Save and export your results to an HDF5 file named AlphabetSoupCharity_Optimization.h5 (1 point) 
#### Write a Report on the Neural Network Model (30 points) 
Write an analysis that includes a title and multiple sections, labeled with headers and subheaders (4 points) <br>
Format images in the report so that they display correction (2) <br>
Explain the purpose of the analysis (4) <br>
Answer all 6 questions in the results section (10) <br>
Summarize the overall results of your model (4) <br>
Describe how you could use a different model to solve the same problem, and explain why you would use that model (6) 
