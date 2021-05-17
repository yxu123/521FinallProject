# 521FinallProject
521FinalProject(Runze, Ze, Yuquan)

## Data
https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset?select=True.csv
  (Fake.csv, Real.csv)

https://www.kaggle.com/snap/amazon-fine-food-reviews
  (Reviews.csv)

https://www.kaggle.com/azimulh/tweets-data-for-authorship-attribution-modelling?select=tweet.csv
  (tweet.csv)

## Third-party libraries necessary to run the code
pandas, numpy, gensim, nltk, string, argparse, sklearn, feature_extraction

## Usage for Scripts
data_clean.py: We cleaned the data and created labels so the data can suit the Naive Bayes and the Logistic Regression. 

feature_extraction.py: Functions we are going to use to parse the datasets, split data into training and test set, and convert labels to a vector y, mapped to 0 and 1

model.py: Train and evaluate the Naive Bayes model and the Logistic Regression model. And compare their performance on three datasets. 
 

## Command line arguments
To run the models on Amazon fine food reviews dataset: python model.py --data_file review.csv 

To run the models on Tweets dataset: python model.py --datafile new_tweet.csv

To run the models on Fake and real news dataset: python model.py --datafile new.csv
