# 521FinallProject
521FinalProject(Runze, Ze, Yuquan)

https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset?select=True.csv
  (Fake.csv, Real.csv)

https://www.kaggle.com/snap/amazon-fine-food-reviews
  (Reviews.csv)

https://www.kaggle.com/azimulh/tweets-data-for-authorship-attribution-modelling?select=tweet.csv
  (tweet.csv)

third-party libraries necessary to run the code：pandas, numpy, gensim, nltk, string, argparse, sklearn, feature_extraction


data_clean.py: We cleaned the data and created labels so the data can suit the Naive Bayes and the Logistic Regression. 

feature_extraction.py: Functions we are going to use to parse the datasets, split data into training and test set, and convert labels to a vector y, mapped to 0 and 1

model.py: Train and evaluate the Naive Bayes model and the Logistic Regression model. And compare their performance on three datasets. 

models_NB.py: models_NB.py will calculate the Accuracy, recall rate, precision and the f1_score for the dataset you have selected. To select the dataset, delete the "#" before the parser.add_argument, like:parser.add_argument("--data_file", type=str, default="review.csv", help="dev file"). Make sure the model_NB.py, feature_extraction.py and all the datasets are in the same directory. 

To run the models on Amazon fine food reviews dataset: python model.py --data_file review.csv 
To run the models on Tweets dataset: python model.py --datafile new_tweet.csv
To run the models on Fake and real news dataset: python model.py --datafile new.csv
