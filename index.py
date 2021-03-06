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

"""
# content = "hai how are you, how its goning in home. Saying is hai is better than saying bye!"
# print content

words_in_content = word_tokenize(content)
# print words_in_content
sentence_in_content = sent_tokenize(content)
# print sentence_in_content

stopwords = stopwords.words('english')
# print stopwords

filtered_words_in_content = [ word.lower() for word in words_in_content if word.lower() not in stopwords ]
print filtered_words_in_content

words_frequency_in_content = FreqDist(filtered_words_in_content)
featured_words = words_frequency_in_content.keys()

def extract_features(words_from_sentence):
	global featured_words
	# result = dict()
	# for word in words_from_sentence:
	# 	result[ 'contains(%s)' % word ] = (word in featured_words)
	# return result
	# return dict(('contains(%s)' % word, (word in featured_words)) for word in words_from_sentence )
	return {'contains(%s)' % word: (word in featured_words) for word in words_from_sentence }


pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(words_frequency_in_content.items())
# pp.pprint(featured_words)
# pp.pprint( extract_features(filtered_words_in_content) )

training_set = nltk.classify.apply_features(extract_features, filtered_words_in_content)

print(training_set)

"""
