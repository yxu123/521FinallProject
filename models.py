# #!/usr/bin/env python
import argparse
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import nltk
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from feature_extraction import *
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
#
#
#
#
def main(data_file):
    labels,texts=parse(data_file)
    labels_map = labels_to_key(labels)
    y = np.asarray(labels_to_y(labels, labels_map))

    train, test = split_data(texts, y, test_percent=0.25, shuffle=True)

    vectorizer = TfidfVectorizer("content", lowercase=True, analyzer="word",
                                         use_idf=True, min_df=10,stop_words=stopwords.words('english'))

    train_x=vectorizer.fit_transform(train[0])
    train_y=train[1]

    #test_vectorizer=TfidfVectorizer("content", lowercase=True, analyzer="word",
    #                                     use_idf=True, min_df=10,stop_words=stopwords.words('english'))
    test_x=vectorizer.transform(test[0])
    test_y=test[1]

    print("---train_x.shape",train_x.shape,"---train_y.shape", train_y.shape)
    print("---test_x.shape", test_x.shape,"---test_y.shape", test_y.shape)

    #Naive Bayes
    m_nb_model = MultinomialNB()
    m_nb_model.fit(train_x, train_y)
    #accuracy
    test_accuracy = m_nb_model.score(test_x, test_y)
    print(f"Accuracy of multinomial NB: {test_accuracy:0.03f}")
    #predict
    predictions = m_nb_model.predict(test_x)
        ###calculate accuracy_score
    print("accuracy_score:", accuracy_score(test_y, predictions))
        ###calculate recall_score
    print("recall_score:",recall_score(test_y, predictions))
        ###calculate precision_score
    print("precision_score:",precision_score(test_y, predictions))
        ###calculate f1_score
    print("F1_score：",f1_score(test_y, predictions))

    # Train a logistic regression model using sklearn.linear_model.LogisticRegression
    print("**********Logistic Regression**********")
    lrc = LogisticRegression()
    lrc.fit(train_x, train_y)
    y_pred = lrc.predict(test_x)
    print(f"accuracy: {accuracy_score(test_y, y_pred):.3f}")
    print(f"precision: {precision_score(test_y, y_pred):.3f}")
    print(f"recall: {recall_score(test_y, y_pred):.3f}")
    print(f"F1 score: {f1_score(test_y, y_pred):.3f}")
    lr_probs = lrc.predict_proba(test_x)[:, 1]
    lr_auc = roc_auc_score(test_y, lr_probs)
    print(f"ROC AUC: {lr_auc}")
    print("\n")



    # Baseline
    # predicts the class value that has the most observations in the training dataset
    print("**********Zero-rule**********")
    predict = zero_rule_algorithm_classification(train_y, test_y)
    print(f"accuracy: {accuracy_score(test_y, predict):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--data_file", type=str, default="review.csv",
    #                    help="dev file")
    #parser.add_argument("--data_file", type=str, default="new_tweet.csv",
    #                    help="dev file")
    parser.add_argument("--data_file", type=str, default="news.csv", help="dev file")

    args = parser.parse_args()

    main(args.data_file)
