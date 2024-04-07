# Email Spam Detection Project

## Introduction
This project aims to classify emails into 'spam' or 'ham' (non-spam) using machine learning techniques. We employ a Naive Bayes classifier in conjunction with advanced text preprocessing methods including normalization, tokenization, stopword removal, stemming, and lemmatization, as well as feature extraction with CountVectorizer and TfidfTransformer.

## Environment Setup
To run this project, you need Python and the following libraries:
- pandas
- numpy
- nltk
- scikit-learn



## Dataset
You can download the data set from: https://www.kaggle.com/datasets/venky73/spam-mails-dataset/data

## Preprocessing Steps
1. **Normalization**: Convert text to lowercase and remove non-alphanumeric characters.
2. **Tokenization**: Split text into individual words.
3. **Stopword Removal**: Remove common words that are unlikely to contribute to the classification.
4. **Stemming and Lemmatization**: Reduce words to their root or base form.
5. **Feature Extraction**: Use CountVectorizer with bi-grams and TfidfTransformer for converting text to numeric form.

## Model
We use the Multinomial Naive Bayes classifier with an alpha parameter of 0.01. The model is trained on TF-IDF transformed text features.

## Results
The model achieved the following performance on the test set:

              precision    recall  f1-score   support

         ham       0.99      0.99      0.99      1121
        spam       0.97      0.98      0.97       431

    accuracy                           0.98      1552
   macro avg       0.98      0.98      0.98      1552
weighted avg       0.98      0.98      0.98      1552

These results indicate that the model is highly effective at distinguishing between spam and ham emails, with a high degree of precision and recall for both classes.
