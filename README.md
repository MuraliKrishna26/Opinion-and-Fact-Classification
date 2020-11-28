# Opinion-and-Fact-Classification

## Problem Definition

In the present-day technology huge amount of data is being generated every day. So, it’s turning out to be a challenging task to handle text-based data. In the world of text-based sentences it is not that simple to differentiate between fact and opinions. So, our project is to build the model that classifies/identifies facts from/and opinions in the given text by using various machine learning and deep learning techniques.

## Dataset Used

The dataset we will be using for this project is hand annotated. We considered the data from “movies” domain and annotated them into opinions and facts. Here, the plot of a movie is considered as fact. whereas the review of an individual for a movie is considered as opinion. The dataset contains 94,379 samples which are facts or opinions. Dataset has opinion count of 50,000 whereas facts of 44,379. The dataset has train, cross-validation & test splits.
We this available data we propose to classify opinions from facts using various machine learning and deep learning techniques.

## Data Pre-Processing

    - Stop-Word removal
    - Case Conversion
    - Tokenization, Lemmatization
    - Removal of alpha-numeric words and special characters.
    - Removal of words of length less than 3.
    
## Feature Extraction 

    - Bag Of Words
    - TF-IDF
    
## Models Used 

    - KNN
    - Naive Bayes
    - Decision Trees
    - SVM
    - LSTM












