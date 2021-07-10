import xml.etree.ElementTree as ET
import os
import nltk
nltk.download('punkt')
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSeq2SeqLM.from_pretrained("opus-mt-en-ROMANCE").to(device)
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained("opus-mt-en-ROMANCE")

def translate(texts):
    try:
        # tokenizando as sentenças
        encoded = tokenizer(texts, return_tensors='pt', padding=True).to(device)
        # traduzindo
        translated = model.generate(**encoded)
        # preparando a saída
        translations = tokenizer.batch_decode(translated, skip_special_tokens=True)
    except:
        print('Translation Error.')
        translations = ['' for w in texts]
    return translations

def parse(path):
    translations = []
    tree = ET.parse(os.path.join(fdir, fname))
    document = tree.getroot()
    for qpair in document.find('QAPairs').findall('QAPair'):
        q = qpair.find('Question')
        qid = q.attrib['qid']
        qtype = q.attrib['qtype']
        question = q.text

        q_snts = ['>>pt_br<< ' + w for w in nltk.sent_tokenize(question)]
        q_trans = ' '.join(translate(q_snts))

        a = qpair.find('Answer')
        answer = a.text
        ans_trans = ''
        if answer:
            ans_snts = ['>>pt_br<< ' + w for w in nltk.sent_tokenize(answer)]
            ans_trans = ' '.join(translate(ans_snts))

        qa = { 'qid': qid, 'qtype': qtype, 'question': q_trans, 'answer': ans_trans }
        translations.append(qa)
    return translations

if not os.path.exists('translations'):
    os.mkdir('translations')

for fdir in os.listdir('.'):
    if fdir.split('_')[0].isdigit():
        if not os.path.exists(os.path.join('translations', fdir)):
            os.mkdir(os.path.join('translations', fdir))
        for fname in os.listdir(fdir):
            path = os.path.join(fdir, fname)
            try:
                fwrite = os.path.join('translations', fdir, fname.replace('.xml', '.json'))
                print(path)
                if not os.path.exists(fwrite):
                    translations = parse(path)
                    json.dump(translations, open(fwrite, 'w'), separators=(',', ':'), indent=4)
            except:
                print(path)
