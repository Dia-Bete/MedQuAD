__author__='thiagocastroferreira'

import csv
import json

from bm25 import BM25QA
from tfidf_cosine import TFIDFCosineQA
from softcosine import TFIDFSoftCosineQA
from muse import MUSEQA
from bert import BERTQA

def load_questions(path):
    teste = json.load(open(TESTE_PATH))
    return teste

if __name__ == "__main__":
    TESTE_PATH = '../QA/medicine/test.json'

    CORPORA_PATH = [
        '../QA/adalberto/qa_pt.json',
        '../QA/diabetes_action/qa_pt.json',
        '../QA/diabetesbr/qa_pt.json',
        '../QA/eatingwell/qa_pt.json',
        '../QA/medicine/qa_pt.json'
    ]

    teste = load_questions(TESTE_PATH)

    results = []

    # BM25
    bm25 = BM25QA(CORPORA_PATH)
    for qa in teste:
        q = qa['question']
        bm25_candidates = bm25.score(q)

        result = { 
            'question': qa['question'], 
            'bm25': ""
        }
        if len(bm25_candidates) > 0:
            result['bm25'] = bm25_candidates[0]['a']
    del bm25

    # Cosine
    tfidfcosine = TFIDFCosineQA(CORPORA_PATH)
    for i, qa in enumerate(teste):
        q = qa['question']
        cosine_candidates = tfidfcosine.score(q)
        
        if len(cosine_candidates) > 0:
            results[i]['tfidf_cosine'] = cosine_candidates[0]['a']
    del tfidfcosine
    
    # Softcosine
    tfidfsoftcosine = TFIDFSoftCosineQA(CORPORA_PATH)
    tfidfsoftcosine.train()
    for i, qa in enumerate(teste):
        q = qa['question']
        softcosine_candidates = tfidfsoftcosine.score(q)
        if len(softcosine_candidates) > 0:
            results[i]['tfidf_softcosine'] = softcosine_candidates[0]['a']
    del tfidfsoftcosine
    
    # MUSE
    muse = MUSEQA(CORPORA_PATH)
    muse.train()
    for i, qa in enumerate(teste):
        q = qa['question']
        muse_candidates = muse.score(q)
        if len(muse_candidates) > 0:
            results[i]['muse'] = muse_candidates[0]['a']
    del muse
    
    # BERT
    bert = BERTQA(CORPORA_PATH)
    bert.train()
    for i, qa in enumerate(teste):
        q = qa['question']
        bert_candidates = bert.score(q)
        if len(bert_candidates):
            result['bert'] = bert_candidates[0]['a']
        results.append(result)
    del bert

    # Siamese

    with open('candidates.csv', 'w') as f:
        writer = csv.writer(f, delimiter=';', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["Question", "A1: BM25", "A2: TF-IDF Cosine", "A3: TF-IDF Cosine", "A4: MUSE", "A5: BERT"])
        for result in results:
            writer.writerow([
                result['question'],
                result['bm25'],
                result['tfidf_cosine'],
                result['tfidf_softcosine'],
                result['muse'],
                result['bert']
            ])