import argparse
from collections import defaultdict
import glob
import json
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import ntpath
import os
import sys

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords


def find_frequent_words(documents_dir, select_count, output_file):
	# set up our nltk tools
	stemmer = SnowballStemmer('english')
	stop_words = set(stopwords.words('english'))
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	documents = []
	# all sentences
	sentences = []
	# map sentences to documents
	sentence_document_map = []
	word_frequency = defaultdict(int)
	# map words to sentences
	word_sentence_map = defaultdict(set)
	# map original form of words to their stems
	stemmed_word_map = defaultdict(set)

	# loop documents in documents dir
	for index, file in enumerate(glob.glob(os.path.join(documents_dir, '*'))):
		documents.append(ntpath.basename(file))
		with open(file, 'r', encoding='utf-8') as f:
			# starting sentence index for current document
			num_sentences = len(sentences)
			content = f.read()
			content_sentences = tokenizer.tokenize(content)
			num_new_sentences = len(content_sentences)
			# add new sentences to all sentences list
			sentences.extend(content_sentences)
			# map new sentences to current document
			sentence_document_map.extend([index] * len(content_sentences))

			# process sentences into words and stems
			# alternately, we can use scikit learn count vectorizer here
			for i in range(num_new_sentences):
				# get index of sentence in all sentences list
				sentence_index = i + num_sentences
				words = word_tokenize(content_sentences[i].lower())
				for w in words:
					if w.isalpha() and w not in stop_words:
						stemmed = stemmer.stem(w)
						word_frequency[stemmed] += 1
						# map word (stem) to sentence
						word_sentence_map[stemmed].add(sentence_index)
						# track original forms of the stemmed word
						stemmed_word_map[stemmed].add(w)

	wf_sorted = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)

	# frequent words and details
	frequent_words = []
	# one list of associated sentences
	sentence_references = {}
	for root, count in wf_sorted[:select_count]:
		details = {
			'root': root,
			'frequency': count,
			'documents': set(),
			'sentences': defaultdict(list),
			'forms': list(stemmed_word_map[root]),
		}
		for sentence_index in word_sentence_map[root]:
			details['documents'].add(sentence_document_map[sentence_index])
			details['sentences'][sentence_document_map[sentence_index]].append(sentence_index)
			sentence_references[sentence_index] = sentences[sentence_index]
		details['documents'] = list(details['documents'])
		details['num_sentences'] = len(details['sentences'])
		frequent_words.append(details)

	results = {
		'frequent_words': frequent_words,
		'documents': documents,
		'sentences': sentence_references,
	}

	with open(output_file, 'w') as f:
		f.write('let data={}'.format(json.dumps(results)))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Find most frequent words in a set of documents.')
	parser.add_argument('--dir', type=str, required=True, help='Documents directory')
	parser.add_argument('--select', type=int, default=20, help='Number of words to select')
	
	args = parser.parse_args()
	documents_dir = args.dir
	select = args.select

	if not os.path.isdir(documents_dir):
		sys.exit('Invalid documents directory')
	if select <= 0:
		sys.exit('Please enter a positive number for word selection')

	find_frequent_words(documents_dir, select, 'data.js')
