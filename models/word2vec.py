__author__='thiagocastroferreira'

import json
import nltk
import os

import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

class Word2VecQA:
    def __init__(self, corpora_paths, word2vec_path, candidate_format='question'):
        """
        Args:
            corpora_paths: path to the training corpora
            word2vec_path: path to word2vec
            candidate_format: should a candidate be represented by its question, answer or question+answer?
        """
        self.corpus = []
        for path in corpora_paths:
            self.corpus.extend(json.load(open(path)))

        self.candidate_format = candidate_format
        self.word2vec = KeyedVectors.load_word2vec_format(word2vec_path)

    def train(self):
        self.retrieval = []
        for i, text in enumerate(self.corpus):
            if self.candidate_format == 'question':
                candidate = text['question']
            elif self.candidate_format == 'answer':
                candidate = text['answer']
            else:
                candidate = ' '.join([text['question'], text['answer']])
            self.retrieval.append(self.embed(candidate))
        self.retrieval = np.array(self.retrieval)


    def embed(self, text):
        texto_tok = nltk.word_tokenize(text, language='portuguese')
        
        q_embs = []
        for w in texto_tok:
            try:
                q_embs.append(self.word2vec[w.lower()])
            except:
                q_embs.append(self.word2vec['oov'])
        return np.mean(q_embs, axis=0)
    

    def score(self, question):
        assert self.retrieval
        q_emb = self.embed(question)
        
        cosine = cosine_similarity([q_emb], self.retrieval)
        scores = list(cosine[0])
        # scores = np.inner(q_emb, self.retrieval)
        
        results, answers = [], []
        for i, s in enumerate(scores):
            if s > 0:
                if self.corpus[i]['answer'] not in answers:
                    answers.append(self.corpus[i]['answer'])
                    answer = self.corpus[i]['answer'].replace('\ n', '\n')
                    results.append({ 'q': self.corpus[i]['question'], 'a': answer, 'index': i, 'score': s })
        return sorted(results, key=lambda x: x['score'], reverse=True)[:5]

    def score_pair(self, q1, q2):
        q1_emb = self.embed(q1)
        q2_emb = self.embed(q2)

        score = cosine_similarity([q1_emb], [q2_emb])[0][0]
        # score = np.inner(q1_emb, q2_emb)
        return score