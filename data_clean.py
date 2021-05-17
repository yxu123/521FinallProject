import pandas as pd
import re, string
from nltk import word_tokenize

# Read csv files as pandas data frames
true=pd.read_csv('True.csv')
fake=pd.read_csv('Fake.csv')

# Add new column to data frames

true['true_or_fake']=['true']*len(true)
fake['true_or_fake']=['fake']*len(fake)

# Combine two data frames
news=pd.concat([true, fake],axis=0)
#news=true.append(fake)

# Drop columns we are not going to use
news=news.drop(['date','title','subject'], axis=1)
# Drop rows which contain missing value
news=news.dropna()


# Escape separator
for i in range(len(news)):
    #news['text'].iloc[i]=" ".join(news['text'].iloc[i].split())
    news['text'].iloc[i]=re.sub(r'[^\w\d]', " ",news['text'].iloc[i])

# Write new dataset to a csv file
news.to_csv('news.csv',sep='\t',index=False,header=False)


# Read csv file as pandas data frame
tweet=pd.read_csv('tweet.csv')
# Drop rows that contain missing value
tweet=tweet.dropna()

# Remove tweets that are not writen by Barack Obama not Neil deGrasse Tyson
tweet=tweet.drop(tweet[(tweet['Name']!='Barack Obama') & (tweet['Name']!='Neil deGrasse Tyson')].index,axis=0)

# Adjust the position of columns
tweet.insert(0, 'texts', tweet['Tweet description'])
tweet=tweet.drop('Tweet description',axis=1)
# Escape separator
for i in range(len(tweet)):
    tweet['texts'].iloc[i] = " ".join(tweet['texts'].iloc[i].split())
    #tweet['texts'].iloc[i] = re.sub(r'[^\w\d]', " ", tweet['texts'].iloc[i])
# Write the output to csv file
tweet.to_csv('new_tweet.csv',sep='\t',index=False,header=False)

# Read csv file as pandas data frame
review=pd.read_csv('Reviews.csv')

# Since the dataset is too large, we decide to extract a sample from it and use it as our data.
review=review.iloc[:10000]

# Drop columns that we are not going to use
review=review.drop(['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator',
                                                                                   'Time','Summary'],axis=1)
# Drop rows that contain missing value
review=review.dropna()

# Add new column to dataset
fivestar = []
for i in range(len(review)):
    if review['Score'].loc[i]==5:
        fivestar.append('Yes')
    else:
        fivestar.append('No')
review['FiveStar']=fivestar
review=review.drop('Score',axis=1)
# Escape separator
for i in range(len(review)):
    review['Text'].iloc[i] = " ".join(review['Text'].iloc[i].split())
    #review['Text'].iloc[i]= re.sub(r'[^\w\d]', " ", review['Text'].iloc[i])
# Write the output to csv file
review.to_csv('review.csv',sep='\t',index=False,header=False)