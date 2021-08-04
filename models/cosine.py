__author__='thiagocastroferreira'

import json
import nltk

stopwords = nltk.corpus.stopwords.words('portuguese')

import string
punctuation = string.punctuation

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='portuguese')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

def tokenize(text):
    return nltk.word_tokenize(text, language='portuguese')

def preprocess(q):
    q = nltk.word_tokenize(q, language='portuguese')
    q = [stemmer.stem(w.lower()) for w in q if w not in stopwords and w not in punctuation]
    return ' '.join(q)

class TFIDFCosineQA:
    def __init__(self, corpora_paths, candidate_format='question'):
        """
        Args:
            corpora_paths: path to the training corpora
            candidate_format: should a candidate be represented by its question, answer or question+answer?
        """
        self.corpus = []
        for path in corpora_paths:
            self.corpus.extend(json.load(open(path)))

        self.candidate_format = candidate_format
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, preprocessor=preprocess)
        
        if self.candidate_format == 'question':
            candidates = [qa['question'] for qa in self.corpus]
        elif self.candidate_format == 'answer':
            candidates = [qa['answer'] for qa in self.corpus]
        else:
            candidates = [' '.join([qa['question'], qa['answer']]) for qa in self.corpus]
        self.retrieval = self.tfidf.fit_transform(candidates)
    
    def score(self, question):
        q_tfidf = self.tfidf.transform([question])
        
        cosine = cosine_similarity(q_tfidf, self.retrieval)
        scores = list(cosine[0])
        
        results, answers = [], []
        for i, s in enumerate(scores):
            if s > 0:
                if self.corpus[i]['answer'] not in answers:
                    answers.append(self.corpus[i]['answer'])
                    answer = self.corpus[i]['answer'].replace('\ n', '\n')
                    results.append({ 'q': self.corpus[i]['question'], 'a': answer, 'index': i, 'score': s })
        return sorted(results, key=lambda x: x['score'], reverse=True)[:5] 

    def score_pair(self, q1, q2):
        q_tfidf = self.tfidf.transform([q1, q2])
        q1_emb, q2_emb = q_tfidf[0], q_tfidf[1]
        score = cosine_similarity(q1_emb, q2_emb)[0][0]
        return score