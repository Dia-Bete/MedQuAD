__author__='thiagocastroferreira'

import json
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import torch
from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModelForPreTraining  # Or BertForPreTraining for loading pretraining heads
from transformers import AutoModel  # or BertModel, for BERT without pretraining heads

class BERTQA:
    def __init__(self, corpora_paths, pretrained_tokenizer='neuralmind/bert-large-portuguese-cased', pretrained_model='neuralmind/bert-large-portuguese-cased', candidate_format='question'):
        """
        Args:
            corpora_paths: path to the training corpora
            pretrained_tokenizer: path to tokenizer
            pretrained_model: path to the QA model
            candidate_format: should a candidate be represented by its question, answer or question+answer?
        """
        self.corpus = []
        for path in corpora_paths:
            self.corpus.extend(json.load(open(path)))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.candidate_format = candidate_format
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer, do_lower_case=False)
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.bert = self.bert.to(self.device)

    def train(self):
        self.retrieval = []
        for i, text in enumerate(self.corpus):
            if self.candidate_format == 'question':
                candidate = text['question']
            elif self.candidate_format == 'answer':
                candidate = text['answer']
            else:
                candidate = ' '.join([text['question'], text['answer']])
            self.retrieval.append(self.embed(candidate_format))
        self.retrieval = np.array(self.retrieval)


    def embed(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            outs = self.bert(input_ids)
            encoded = outs[0][0, 1:-1]  # Ignore [CLS] and [SEP] special tokens
            return torch.mean(encoded, dim=0).cpu().numpy()
            # encoded = outs[0][0, 0] 
            # return encoded.cpu().numpy()
    

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