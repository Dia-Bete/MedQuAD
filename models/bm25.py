__author__='thiagocastroferreira'

import json
import nltk
from gensim.summarization.bm25 import BM25

palavras_vazias = nltk.corpus.stopwords.words('portuguese')

import string
pontuacao = string.punctuation

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='portuguese')

import numpy as np

def preprocess(texto):
    texto_tok = nltk.word_tokenize(texto, language='portuguese')
    texto_tok = [stemmer.stem(w.lower()) for w in texto_tok if w.lower() not in palavras_vazias and w.lower() not in pontuacao]
    return texto_tok

class BM25QA:
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
        
        # train bm25
        corpus_tok = []
        for q in self.corpus:
            if self.candidate_format == 'question':
                corpus_tok.append(preprocess(q['question']))
            elif self.candidate_format == 'answer':
                corpus_tok.append(preprocess(q['answer']))
            else:
                candidate = preprocess(' '.join([q['question'], q['answer']]))
                corpus_tok.append(candidate)
        self.retrieval = BM25(corpus_tok)

        # clean corpus
        for i, qa in enumerate(self.corpus):
            del self.corpus[i]['category']
            del self.corpus[i]['source']

    def score(self, question):
        text_tok = preprocess(question)
        scores = self.retrieval.get_scores(text_tok)
        
        results, answers = [], []
        for i, s in enumerate(scores):
            if s > 0:
                if self.corpus[i]['answer'] not in answers:
                    answers.append(self.corpus[i]['answer'])
                    answer = self.corpus[i]['answer'].replace('\ n', '\n')
                    results.append({ 'q': self.corpus[i]['question'], 'a': answer, 'index': i, 'score': s })
        return sorted(results, key=lambda x: x['score'], reverse=True)[:5]

    def score_pair(self, q1, q2):
        idx = -1
        for idx, q in enumerate(self.corpus):
            if self.candidate_format == 'question':
                candidate = q['question']
            elif self.candidate_format == 'answer':
                candidate = q['answer']
            else:
                candidate = ' '.join([q['question'], q['answer']])
            
            if q1 == candidate:
                break

        q2_tok = preprocess(q2)
        score = self.retrieval.get_score(q2_tok, idx)
        return score

if __name__ == "__main__":
    TESTE_PATH = '../QA/medicine/test.json'

    CORPORA_PATH = [
        '../QA/adalberto/qa_pt.json',
        '../QA/diabetes_action/qa_pt.json',
        '../QA/diabetesbr/qa_pt.json',
        '../QA/eatingwell/qa_pt.json',
        '../QA/medicine/qa_pt.json'
    ]
        
    bm25 = BM25QA(CORPORA_PATH)

    teste = json.load(open(TESTE_PATH))
    results = []
    for qa in teste:
        q = qa['question']
        candidates = bm25.score(q)
        
        result = {
            'question': qa,
            'candidates': []
        }

        for candidate in candidates:
            result['candidates'].append({
                'question': candidate['q'],
                'answer': candidate['a']
            })
        results.append(result)