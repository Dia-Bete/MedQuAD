__author__='thiagocastroferreira'

import json
import nltk
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import Word2Vec

import torch
from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModelForPreTraining  # Or BertForPreTraining for loading pretraining heads
from transformers import AutoModel  # or BertModel, for BERT without pretraining heads

from sklearn.metrics.pairwise import cosine_similarity

class TFIDFSoftCosineQA:
    def __init__(self, corpora_paths, pretrained_tokenizer='neuralmind/bert-large-portuguese-cased', pretrained_model='neuralmind/bert-large-portuguese-cased'):
        self.corpus = []
        for path in corpora_paths:
            self.corpus.extend(json.load(open(path)))

        document, questions = [], []
        for i, qa in enumerate(self.corpus):
            question = nltk.word_tokenize(qa['question'], language='portuguese')
            self.corpus[i]['question_tok'] = question
            document.append(question)
        
        # TF-IDF
        self.dict = Dictionary(document)
        self.tfidf = TfidfModel([self.dict.doc2bow(line) for line in document])

        # BERT
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer, do_lower_case=False)
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.bert = self.bert.to(self.device)

    def train(self):
        self.retrieval = []
        for i, text in enumerate(self.corpus):
            self.retrieval.append(self.embed(text['question']))

    def embed(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors='pt')
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            outs = self.bert(input_ids)
            wordpieces = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            subwords_idx = [] # first subword of each word
            for i, wordpiece in enumerate(wordpieces):
                if '##' not in wordpiece and i not in [0, len(wordpieces)-1]:
                    subwords_idx.append(i)
                    
            encoded = outs[0][0][subwords_idx]  # Ignore [CLS] and [SEP] special tokens
            return encoded.cpu().numpy()

    def softdot(self, q1, q1_bert, q2, q2_bert):
        # tf-idf
        q1_tfidf = self.tfidf[self.dict.doc2bow(q1)]
        q2_tfidf = self.tfidf[self.dict.doc2bow(q2)]

        cos = 0.0
        for i, w1 in enumerate(q1_tfidf):
            for j, w2 in enumerate(q2_tfidf):
                q1_emb, q2_emb = q1_bert[i], q2_bert[j]
                
                m_ij = max(0, cosine_similarity([q1_emb], [q2_emb])[0][0])**2
                cos += (w1[1] * m_ij * w2[1])
        return cos

    def score(self, question):
        assert self.retrieval
        q1_tok = nltk.word_tokenize(question, language='portuguese')
        q1_bert = self.embed(question)
        
        scores = []
        for i, row in enumerate(self.corpus):
            q2_tok = row['question_tok']
            q2_bert = self.retrieval[i]
            cosine = self.softdot(q1_tok, q1_bert, q2_tok, q2_bert)
            scores.append(cosine)
        
        results, answers = [], []
        for i, s in enumerate(scores):
            if s > 0:
                if self.corpus[i]['answer'] not in answers:
                    answers.append(self.corpus[i]['answer'])
                    answer = self.corpus[i]['answer'].replace('\ n', '\n')
                    results.append({ 'q': self.corpus[i]['question'], 'a': answer, 'index': i, 'score': s })
        return sorted(results, key=lambda x: x['score'], reverse=True)[:5] 

    def score_pair(self, q1, q2):
        q1_tok = nltk.word_tokenize(q1, language='portuguese')
        q1_bert = self.embed(q1)

        q2_tok = nltk.word_tokenize(q2, language='portuguese')
        q2_bert = self.embed(q2)
        score = self.softdot(q1_tok, q1_bert, q2_tok, q2_bert)
        return score
