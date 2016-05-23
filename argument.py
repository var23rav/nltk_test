import pprint

import nltk

from nltk import FreqDist, classify, NaiveBayesClassifier

from nltk.tokenize import word_tokenize, sent_tokenize
# nltk.download("punkt")
from nltk.corpus import stopwords
# nltk.download('stopwords')

positive_contents = [
	('how positive sentense', 'positive'),
	('I agreed to you statement', 'positive'),
	('Sorry for the interuption', 'positive'),
]
negative_contents = [
	('how negative sentense', 'negative'),
	('I can not agree with your statement', 'negative'),
	("I can't agree with your statement", 'negative'),
	("How could you say like that", 'negative'),
	("How could you do that", 'negative'),
	("I didn't do anything", 'negative'),
]
contents = positive_contents + negative_contents

def tokenize_sentence_by_words(contents):
	return [ ( word_tokenize(sentense), status ) for (sentense, status) in contents ]

def get_all_filtered_words_in_sentence(contents):
	from nltk.corpus import stopwords
	filter_word_list = []
	stopwords = stopwords.words('english')	
	for (word_list, status) in tokenize_sentence_by_words(contents):
		filter_word_list.extend( [ word.lower() for word in word_list if word.lower() not in stopwords ] )
	return filter_word_list 

def get_featured_words_in_sentence(contents):
	return FreqDist( get_all_filtered_words_in_sentence(contents) ).keys()
featured_words = get_featured_words_in_sentence(contents)


def extract_features(contents):
	global featured_words
	contents_words = set(contents)
	return { ('contains(%s)' % featured_word ) : (featured_word in contents_words) for featured_word in featured_words }

# Obtaining classifier
# input_contents = [
# 	('positive sentense', 'positive'),
# 	('negative sentense', 'negative'),
# 	('"hai how are you, how its goning in home. Saying is hai is better than saying bye!"', 'negative')
# ]
training_set =  nltk.classify.apply_features(extract_features, tokenize_sentence_by_words(contents) )
print training_set

classifier = nltk.NaiveBayesClassifier.train(training_set)
print classifier

print classifier.show_most_informative_features(32)

tweet = "I didn't do anything like that"
print classifier.classify(extract_features(tweet.split()))