__author__='thiagocastroferreira'

import json
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow_hub as hub
import tensorflow_text

class MUSEQA:
    def __init__(self, corpora_paths, muse_path='muse', candidate_format='question'):
        """
        Args:
            corpora_paths: path to the training corpora
            muse_path: path to MUSE
            candidate_format: should a candidate be represented by its question, answer or question+answer?
        """
        self.corpus = []
        for path in corpora_paths:
            self.corpus.extend(json.load(open(path)))
        self.candidate_format = candidate_format

        self.muse = hub.load(muse_path)
    
    def train(self):
        candidates = []
        for qa in self.corpus:
            if self.candidate_format == 'question':
                candidate = text['question']
            elif self.candidate_format == 'answer':
                candidate = text['answer']
            else:
                candidate = ' '.join([text['question'], text['answer']])
            candidates.append(candidate)
        self.retrieval = self.muse(candidates)

    def score(self, question):
        assert self.retrieval
        q_emb = self.muse(question)
        
        inner = np.inner(q_emb, self.retrieval)
        scores = list(inner[0])
        
        results, answers = [], []
        for i, s in enumerate(scores):
            if s > 0:
                if self.corpus[i]['answer'] not in answers:
                    answers.append(self.corpus[i]['answer'])
                    answer = self.corpus[i]['answer'].replace('\ n', '\n')
                    results.append({ 'q': self.corpus[i]['question'], 'a': answer, 'index': i, 'score': s })
        return sorted(results, key=lambda x: x['score'], reverse=True)[:5]

    def score_pair(self, q1, q2):
        q1_emb = self.muse(q1)
        q2_emb = self.muse(q2)

        score = np.inner(q1_emb, q2_emb)[0][0]
        return score