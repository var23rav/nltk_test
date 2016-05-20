import pprint

import nltk

from nltk import FreqDist, classify

from nltk.tokenize import word_tokenize, sent_tokenize
# nltk.download("punkt")
from nltk.corpus import stopwords
# nltk.download('stopwords')

content = "hai how are you, how its goning in home. Saying is hai is better than saying bye!"
print content

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
	return [ ('contains(%s)' % word, (word in featured_words)) for word in words_from_sentence ]


pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(words_frequency_in_content.items())
# pp.pprint(featured_words)
# pp.pprint( extract_features(filtered_words_in_content) )

training_set = nltk.classify.apply_features(extract_features, filtered_words_in_content)

pp.pprint(training_set)


