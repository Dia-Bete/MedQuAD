__author__='thiagocastroferreira'

import sys
sys.path.append('models')
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch import optim
import metrics as metrics

import csv
import json
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import random

from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModel  # or BertModel, for BERT without pretraining heads
from torch.utils.data import DataLoader, Dataset

class SiameseQA:
    def __init__(self, corpora_paths, model_path, neuralmind_path='neuralmind/bert-large-portuguese-cased', candidate_format='question'):
        self.corpus = []
        for path in corpora_paths:
            self.corpus.extend(json.load(open(path)))

        self.candidate_format = candidate_format
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = Siamese(1024, 2, neuralmind_path)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()


    def train(self):
        self.retrieval = []
        for i, text in enumerate(self.corpus):
            if self.candidate_format == 'question':
                candidate = text['question']
            elif self.candidate_format == 'answer':
                candidate = text['answer']
            else:
                candidate = ' '.join([text['question'], text['answer']])
            self.retrieval.append(self.model.embed(candidate))
        self.retrieval = np.array(self.retrieval)
    

    def score(self, question):
        assert self.retrieval
        q_emb = self.model.embed(question)
        
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
        q1_emb = self.model.embed(q1)
        q2_emb = self.model.embed(q2)

        score = cosine_similarity([q1_emb], [q2_emb])[0][0]
        # score = np.inner(q1_emb, q2_emb)
        return score

def load_data(path):
    data = {}
    with open(path) as f:
        reader = csv.reader(f, delimiter=',', quotechar='\"')
        for row in list(reader)[1:]:
            q1, q2, answer, label, _ = row
            if q1 not in data:
                data[q1] = []
            data[q1].append({
                'q1': q1.strip(),
                'q2': q2.strip(),
                'answer': answer,
                'label': -1 if label.strip().lower() == 'i' else 1
                # 'label': 2 if label == 'S' else 1 if label == 'R' else 0
            })
    return data

def split_data(data, test_size=0.1):
    if not os.path.exists('data.json'):
        user_questions = list(data.keys())
        random.shuffle(user_questions)
        size = int(len(user_questions) * test_size)
        train_q, test_q = user_questions[size:], user_questions[:size]

        traindata = {}
        for i, q in enumerate(train_q):
            traindata[q] = data[q]

        testdata = {}
        for q in test_q:
            testdata[q] = data[q]

        json.dump({ 'train': traindata, 'test': testdata }, open('data.json', 'w'))
    else:
        data_ = json.load(open('data.json'))
        traindata, testdata = data_['train'], data_['test']
    return traindata, testdata

class ProcDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (string): data
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Siamese(nn.Module):
    def __init__(self, tokenizer_path='neuralmind/bert-large-portuguese-cased', model_path='neuralmind/bert-large-portuguese-cased'):
        super(Siamese, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
        self.bert = AutoModel.from_pretrained(model_path)
        self.bert.to(self.device)
    

    def embed(self, q):
        input_ids = self.tokenizer.encode(q, return_tensors='pt', truncation=True, max_length=128).to(self.device)

        with torch.no_grad():
            outs = self.bert(input_ids)
            # encoded = outs[0][0, 1:-1]  # Ignore [CLS] and [SEP] special tokens
            # return torch.mean(encoded, dim=0).cpu().numpy()
            encoded = outs[0][0, 0]  # Ignore [CLS] and [SEP] special tokens
            return encoded.cpu().numpy()
    

    def forward(self, q1, q2):
        token_ids = self.tokenizer(q1, return_tensors='pt', padding=True, truncation=True, max_length=128)
        input_ids = token_ids['input_ids'].to(self.device)
        vector = self.bert(input_ids)
        q1_emb = vector['last_hidden_state'][:, 0, :]
        
        token_ids = self.tokenizer(q2, return_tensors='pt', padding=True, truncation=True, max_length=128)
        input_ids = token_ids['input_ids'].to(self.device)
        vector = self.bert(input_ids)
        q2_emb = vector['last_hidden_state'][:, 0, :]

        return q1_emb, q2_emb


    def distance(self, query, candidates):
        with torch.no_grad():
            query_ = query.detach().cpu().numpy()
            candidates_ = candidates.detach().cpu().numpy()
            scores = cosine_similarity([query_], candidates_)[0]
            # scores = np.inner(query_, candidates_)[0]
            return scores


def prepare_data(trainset, candidate_format='question'):
    result = []
    for qa in traindata:
        q1 = qa['question']
        for i, row in enumerate(qa['candidates']):
            if candidate_format == 'question':
                q2 = row['question']
            elif candidate_format == 'answer':
                q2 = row['answer']
            else:
                q2 = ' '.join([row['question'], row['answer']])
            label = -1 if row['label'].lower().strip() == 'i' else 1
            result.append({ 'q1': q1, 'q2': q2, 'label': label })
    return ProcDataset(result)


def train(model, traindata, criterion, optimizer, device='cuda'):
    torch.cuda.empty_cache()
    model.train()

    losses = []

    batch_candidates, batch_y = [], []
    for batch_idx, qa in enumerate(traindata):
        q1, q2, label = qa['q1'], qa['q2'], qa['label']
            
        # Forward
        q1_emb, q2_emb = model(q1, q2)

        # Calculate loss
        loss = criterion(q1_emb, q2_emb, label.to(device))
        losses.append(float(loss))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Display
        if (batch_idx+1) % 16 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTotal Loss: {:.6f}'.format(
                        epoch, batch_idx + 1, len(traindata),
                               100. * batch_idx / len(traindata), float(loss), round(sum(losses) / len(losses), 5)))
        
    return round(sum(losses) / len(losses), 5)

def test(model, testdata, candidate_format='question'):
    model.eval()
    y_pred, y_real, results = [], [], {}
    for query_idx, qa in enumerate(testdata):
        batch_candidates, batch_y = [], []
        qid, q1 = qa['qid'], qa['question']
        for i, row in enumerate(qa['candidates']):
            if candidate_format == 'question':
                q2 = row['question']
            elif candidate_format == 'answer':
                q2 = row['answer']
            else:
                q2 = ' '.join([row['question'], row['answer']])
            label = -1 if row['label'].lower().strip() == 'i' else 1
            
            batch_candidates.append(q2)
            batch_y.append(label)
            
        q1_emb, q2_emb = model(len(batch_candidates) * [q1], batch_candidates)

        scores = model.distance(q1_emb[0], q2_emb)
        results[qid] = [("false", float(score)) if batch_y[i] == -1 else ("true", float(score)) for i, score in enumerate(scores)]

    return results

if __name__ == "__main__":
    annotatedcv = json.load(open('../QA/annotated/cv.json'))

    for candidate_format in ['question', 'answer', 'question+answer']:
        num_epochs = 30 # The number of epochs (full passes through the data) to train for
        early_stop = 10
        batch_size = 4
        learning_rate = 1e-5
        hidden_dim, num_classes = 1024, 2

        
        tokenizer_path = 'neuralmind/bert-large-portuguese-cased'
        model_path = 'neuralmind/bert-large-portuguese-cased'
        PATH = "model_cosine.pt"

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        results, foldresults, ranking = {}, {}, {}
        map_, mrr = 5 * [0], 5 * [0]
        for fold in annotatedcv:
            print()
            print('Fold: ', fold)
            traindata = annotatedcv[fold]['train']
            traindata = DataLoader(prepare_data(traindata, candidate_format), batch_size=batch_size, shuffle=True)
            
            testdata = annotatedcv[fold]['test']
            ranking[fold] = {}

            model = Siamese(tokenizer_path, model_path)
            model.to(device)

            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
            criterion = nn.CosineEmbeddingLoss(margin=0.5)

            mAP, max_map, repeat = 0, 0, 0
            test(model, testdata, candidate_format)
            for epoch in range(1, num_epochs + 1):
                loss = train(model, traindata, criterion, optimizer)
                foldresults = test(model, testdata, candidate_format)

                for qid in foldresults:
                    ranking[fold][qid] = [w[0] for w in sorted(foldresults[qid], key=lambda x: x[1], reverse=True)]
                
                mAP = metrics.map(ranking[fold], 5)
                print('Fold:', fold, '/ Epoch:', epoch, 'mAP: ', mAP)
                if mAP > max_map:
                    print('Best Model...')
                    results[fold] = foldresults
                    map_[int(fold)-1] = mAP
                    mrr[int(fold)-1] = metrics.mrr(ranking[fold], 5)
                    max_map = mAP
                    repeat = 0
                else:
                    repeat += 1

                if repeat == early_stop:
                    break
        
        path = candidate_format + '_results.json'
        json.dump(results, open(path, 'w'))
        for fold, mAP in enumerate(map_):
            print(fold+1, "Siamese MAP:", round(map_[fold], 2), '/ MRR:', round(mrr[fold], 2))
            print('-')
        print('T Siamese MAP:', round(np.mean(map_), 2), '/ MRR:', round(np.mean(mrr), 2))