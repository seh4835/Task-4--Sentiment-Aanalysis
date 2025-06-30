# Task-4: Sentiment Aanalysis

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SEHER SANGHANI

*INTERN ID*: CT08DL515

*DOMAIN*: DATA ANALYSIS

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH

📌 Project Overview

This repository contains my implementation of sentiment analysis on Twitter reviews using Natural Language Processing (NLP) techniques. As part of my data science and machine learning learning path, I wanted to work on a real-world-like dataset with textual data, to practice cleaning, vectorizing, training, and extracting insights from it.

In this project, I created a sample dataset of 100 synthetic Twitter reviews labelled as positive, negative, and neutral sentiments. I then developed a complete Jupyter notebook pipeline showcasing:

Data loading and exploration

Data cleaning and preprocessing

Text vectorization with CountVectorizer

Model building using Naive Bayes classifier

Evaluation using classification report and confusion matrix

Generating final insights and sample predictions

My main goal was to demonstrate a clear end-to-end NLP workflow for sentiment classification on short text data, especially tweets, which often have casual, informal language and require thorough cleaning.

💡 Motivation
I have always found social media analysis interesting, especially sentiment analysis on tweets. Twitter is widely used by brands, public figures, and regular users to express opinions. Sentiment analysis allows us to extract meaningful trends from these opinions, enabling decision making in marketing, brand reputation management, public opinion analysis, and more.

Through this project, I aimed to:

Strengthen my understanding of NLP preprocessing techniques

Practice applying machine learning models to textual data

Build a reusable notebook template for future NLP tasks

Understand challenges in data cleaning for short texts

Improve my ability to write clean, modular, and well-commented code

🛠️ Technologies Used
Python 3.x

Pandas for data manipulation

NumPy for numerical operations

NLTK for text preprocessing and stopword removal

Scikit-learn for CountVectorizer, model building, evaluation

Matplotlib & Seaborn for data visualization

📂 Dataset
The dataset used is a synthetic dataset created by me containing 100 entries with three sentiment classes:

Positive

Negative

Neutral

Each entry includes:

id: Unique identifier

tweet: The review text

sentiment: Labelled sentiment

Although it is synthetic, I designed the tweets to resemble realistic short reviews to simulate a real Twitter dataset.

📑 Code Explanation
1. Data Loading and Exploration
I load the CSV file using pandas and inspect basic statistics, number of entries per class, and sample rows to understand the dataset.

2. Data Preprocessing
I clean the tweets by:

Converting to lowercase

Removing URLs, mentions, hashtags, and punctuation

Removing stopwords using NLTK’s stopword corpus

This ensures the model focuses on meaningful words in tweets.

3. Vectorization
I use CountVectorizer to convert text data into numerical feature vectors suitable for machine learning models.

4. Model Implementation
I train a Multinomial Naive Bayes classifier, which works effectively for text classification tasks like spam detection and sentiment analysis.

5. Evaluation
I evaluate the model using:

Accuracy Score

Classification Report with precision, recall, and f1-score for each class

Confusion Matrix plotted using Seaborn heatmap for visual analysis of misclassifications

6. Sample Predictions
I test the model on a few new sample tweets to see its predictions on unseen data.

📊 Results and Insights
The model performed well on the small synthetic dataset with high accuracy.

Positive and negative tweets were easier to classify compared to neutral tweets, indicating the model effectively learns strong emotional cues.

Neutral tweets often lack clear sentiment words, hence may require additional feature engineering or advanced models for better classification.

🎯 Future Work
In future iterations, I plan to:

Use real Twitter datasets via Twitter API for authentic data

Implement TF-IDF Vectorizer for better feature representation

Experiment with advanced models such as Logistic Regression, SVM, and LSTM-based deep learning models

Perform hyperparameter tuning to improve performance

Deploy as a web app for interactive sentiment analysis

📌 Final Note
Thank you for visiting this repository! If you found it useful, please ⭐ star it and share your feedback. Your suggestions will help me improve and work on more advanced projects.
