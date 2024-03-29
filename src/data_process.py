import codecs
import sys
import operator
import math
import re
import numpy as np
from collections import defaultdict
import random

def update_vocab(symbol, idxvocab, vocabxid):
    idxvocab.append(symbol)
    vocabxid[symbol] = len(idxvocab) - 1

def gen_vocab(dummy_symbols, corpus, stopwords, vocab_minfreq, vocab_maxfreq):
    idxvocab = []
    vocabxid = defaultdict(int)
    vocab_freq = defaultdict(int)
    for line_id, line in enumerate(codecs.open(corpus, "r", "utf-8")):
        for word in line.strip().split():
            vocab_freq[word] += 1

    for s in dummy_symbols:
        update_vocab(s, idxvocab, vocabxid)


    for w, f in sorted(vocab_freq.items(), key=operator.itemgetter(1), reverse=True):

        if f < vocab_minfreq:
            break
        else:
            update_vocab(w, idxvocab, vocabxid)


    stopwords = set([item.strip().lower() for item in open(stopwords)])
    freqwords = set([item[0] for item in sorted(vocab_freq.items(), key=operator.itemgetter(1), \
        reverse=True)[:int(float(len(vocab_freq))*vocab_maxfreq)]]) 
    alpha_check = re.compile("[a-zA-Z]")
    symbols = set([ w for w in vocabxid.keys() if ((alpha_check.search(w) == None) or w.startswith("'")) ])
    ignore = stopwords | freqwords | symbols | set(dummy_symbols) | set(["n't"])    
    ignore = set([vocabxid[w] for w in ignore if w in vocabxid])

    return idxvocab, vocabxid, ignore

def gen_data(tm_vocab, doc_sents_num, corpus):
    sents = []
    sents_bow=[]
    sent_num = 0
    doc = []
    doc_bow = np.zeros([len(tm_vocab)])
    docs = []
    docs_bow=[]
    tm_voctoid = str_to_id(tm_vocab)
    for line_id, line in enumerate(codecs.open(corpus, "r", "utf-8")):
        for s in line.strip().split("\t"):
            sent = []
            for w in s.strip().split():
                if w in tm_vocab:
                    sent.append(tm_voctoid[w])
            if len(sent)<6:
                break
            sents.append(sent)
            sents_bow.append(bow_sent(sent,len(tm_vocab)))

            doc.append(sent)
            doc_bow += bow_sent(sent,len(tm_vocab))
            sent_num += 1
            if sent_num==doc_sents_num:
                docs.append(doc)
                docs_bow.append(doc_bow)
                doc=[]
                sent_num=0
                doc_bow = np.zeros([len(tm_vocab)])

    return sents,sents_bow, docs,docs_bow

def str_to_id(vocab):
    voctoid={}
    for index,item in enumerate(vocab):
        voctoid[item] = index
    return voctoid

def gen_data_bpe(enc,doc_sents_num, corpus):
    vocab_len=len(enc.encoder)
    sents = []
    sents_bow=[]
    sent_num = 0
    doc = []
    doc_bow = np.zeros([vocab_len])
    docs = []
    docs_bow=[]
    with open(corpus) as f:
        lines = f.readlines()
        lines = lines[:150000]
    for line_id, line in enumerate(lines):
    #for line_id, line in enumerate(codecs.open(corpus, "r", "utf-8")):
        for s in line.strip().split("\t"):
            sent = enc.encode(s)
            if len(sent)<15:   ## for wiki2 and wiki103 : <15
                break
            sents.append(sent)
            sents_bow.append(bow_sent(sent,vocab_len))
            doc.append(sent)
            doc_bow += bow_sent(sent,vocab_len)
            sent_num += 1
            if sent_num==doc_sents_num:
                docs.append(doc)
                docs_bow.append(doc_bow)
                doc=[]
                sent_num=0
                doc_bow = np.zeros([vocab_len])
    return sents,sents_bow, docs,docs_bow


def gen_data_abi(tm_vocab, doc_sents_num, corpus):
    sents = []
    sents_bow=[]
    doc = []
    doc_bow = np.zeros([len(tm_vocab)])
    docs = []
    docs_bow=[]
    tm_voctoid = str_to_id(tm_vocab)
    for line_id, line in enumerate(codecs.open(corpus, "r", "utf-8")):
        for s in line.strip().split("\t"):
            sent = []
            for w in s.strip().split():
                if w in tm_vocab:
                    sent.append(tm_voctoid[w])
            # if len(sent)<5:
            #     continue
            # sents.append(sent)
            # sents_bow.append(bow_sent(sent,len(tm_vocab)))

            # doc.append(sent)
            doc_bow += bow_sent(sent,len(tm_vocab))
        if sum(doc_bow)<20:
            continue
        if line_id % 100 == 0:
            print(str(line_id) + " processed\r")

        docs_bow.append(doc_bow)
        doc_bow = np.zeros([len(tm_vocab)])

    return sents,sents_bow, docs,docs_bow

def Get_TM_vocab(idxvocab,tm_ignore):
    TM_vocab = np.delete(idxvocab, list(tm_ignore))
    tm_vocab = TM_vocab.tolist()
    return tm_vocab


def bow_sent(x, V):
    train_bow = np.zeros([V])
    for word in x:
        train_bow[word] += 1
    return train_bow


def Bag_of_words(TM_original_train,idxvocab,tm_ignore):  
    TM_doc = []
    for d in range(len(TM_original_train)):
        docw = TM_original_train[d]
        docw = pad([item for sublst in docw for item in sublst][:cf.doc_len], cf.doc_len, 0)
        TM_doc.append(docw)
    TM_train_bow = np.zeros([len(idxvocab), len(TM_doc)])
    for doc_index in range(len(TM_doc)):
        for word in TM_doc[doc_index]:
            TM_train_bow[word][doc_index] += 1
    TM_train_bow = np.delete(TM_train_bow, list(tm_ignore), axis = 0)
    return TM_doc, TM_train_bow




def init_embedding(model, idxvocab):
    word_emb = []
    for vi, v in enumerate(idxvocab):
        if v in model:
            word_emb.append(model[v])
        else:
            word_emb.append(np.random.uniform(-0.5/model.vector_size, 0.5/model.vector_size, [model.vector_size,]))
    return np.array(word_emb)

def pad(lst, max_len, pad_symbol):
    return lst + [pad_symbol] * (max_len - len(lst))

def get_batches(sents,Theta,pad_id = 0):
    lm_num_batches =  int(math.ceil(float(len(sents)) / cf.batch_size))
    batch_ids = [item for item in range(lm_num_batches)]
    sent_len = cf.lm_sent_len
    batch_size = cf.batch_size
    random.shuffle(batch_ids)
    random.shuffle(sents)
    X = []
    Y = []
    M = []
    T = []
    for idx in batch_ids:
        x, y, m, t = [], [], [], []
        if idx != max(batch_ids) :
            for docid, seqid, seq in sents[(idx * batch_size):((idx + 1) * batch_size)]:
                x.append(pad(seq[:-1], sent_len, pad_id))
                y.append(pad(seq[1:], sent_len, pad_id))
                m.append([1.0] * (len(seq) - 1) + [0.0] * (sent_len - len(seq) + 1))
                t.append(Theta[:,docid])
        else:
            for docid, seqid, seq in sents[(idx * batch_size):len(sents)]:
                x.append(pad(seq[:-1], sent_len, pad_id))
                y.append(pad(seq[1:], sent_len, pad_id))
                m.append([1.0] * (len(seq) - 1) + [0.0] * (sent_len - len(seq) + 1))
                t.append(Theta[:, docid])
            for padingnum in range((max(batch_ids)+1)*batch_size - len(sents)):
                x.append([pad_id] * sent_len)
                y.append([pad_id] * sent_len)
                m.append([0.0] * sent_len)
                t.append([pad_id] * cf.topic_embedding_size)
        X.append(x)
        Y.append(y)
        M.append(m)
        T.append(t)
    return X,Y,M,T



def get_batches_tm_lm(sents,docbow,V, pad_id = 0 ):
    lm_num_batches = int(math.ceil(float(len(sents)) / cf.batch_size))
    batch_ids = [item for item in range(lm_num_batches)]
    sent_len = cf.lm_sent_len
    batch_size = cf.batch_size
    random.shuffle(batch_ids)
    random.shuffle(sents)
    X = []
    Y = []
    D = []
    M = []
    for idx in batch_ids:
        x, y, d, m = [], [], [], []
        if idx != max(batch_ids):
            for docid, seqid , seq in sents[(idx * batch_size):((idx + 1) * batch_size)]:
                x.append(pad(seq[:-1], sent_len, pad_id))
                y.append(pad(seq[1:], sent_len, pad_id))
                d.append(docbow[: ,docid])
                m.append([1.0] * (len(seq) - 1) + [0.0] * (sent_len - len(seq) + 1))
        else:
            for docid, seqid, seq in sents[(idx * batch_size):len(sents)]:
                x.append(pad(seq[:-1], sent_len, pad_id))
                y.append(pad(seq[1:], sent_len, pad_id))
                d.append(docbow[:, docid])
                m.append([1.0] * (len(seq) - 1) + [0.0] * (sent_len - len(seq) + 1))
            for padingnum in range((max(batch_ids) + 1) * batch_size - len(sents)):
                x.append([pad_id] * sent_len)
                y.append([pad_id] * sent_len)
                d.append(np.array([pad_id] * V))
                m.append([0.0] * sent_len)
        X.append(x)
        Y.append(y)
        D.append(d)
        M.append(m)
    return X, Y, D, M

def get_batches_lm(sents,pad_id = 0):
    lm_num_batches =  int(math.ceil(float(len(sents)) / cf.batch_size))
    batch_ids = [item for item in range(lm_num_batches)]
    sent_len = cf.lm_sent_len
    batch_size = cf.batch_size
    random.shuffle(batch_ids)
    random.shuffle(sents)
    X = []
    Y = []
    M = []
    for idx in batch_ids:
        x, y, m = [], [], []
        if idx != max(batch_ids) :
            for docid, seqid, seq in sents[(idx * batch_size):((idx + 1) * batch_size)]:
                x.append(pad(seq[:-1], sent_len, pad_id))
                y.append(pad(seq[1:], sent_len, pad_id))
                m.append([1.0] * (len(seq) - 1) + [0.0] * (sent_len - len(seq) + 1))
        else:
            for docid, seqid, seq in sents[(idx * batch_size):len(sents)]:
                x.append(pad(seq[:-1], sent_len, pad_id))
                y.append(pad(seq[1:], sent_len, pad_id))
                m.append([1.0] * (len(seq) - 1) + [0.0] * (sent_len - len(seq) + 1))
            for padingnum in range((max(batch_ids)+1)*batch_size - len(sents)):
                x.append([pad_id] * sent_len)
                y.append([pad_id] * sent_len)
                m.append([0.0] * sent_len)
        X.append(x)
        Y.append(y)
        M.append(m)
    return X,Y,M

def get_batches_tm(docbow,pad_id=0.0):
    tm_num_batches = int(math.ceil(float(docbow.shape[1]) / cf.batch_size))
    batch_ids = [item for item in range(tm_num_batches)]
    random.shuffle(batch_ids)
    D = []
    for idx in batch_ids:
        d = []
        if idx != max(batch_ids):
            for docid in range((idx * cf.batch_size),((idx+1) * cf.batch_size)):
                d.append(docbow[: ,docid])
        else:
            for docid in range((idx * cf.batch_size),docbow.shape[1]):
                d.append(docbow[:, docid])
            for padingnum in range((max(batch_ids) + 1) * cf.batch_size - docbow.shape[1]):
                d.append(np.array([pad_id] * docbow.shape[0]))
        D.append(d)
    return D


def get_sent_num(sents,doc_lenth):
    sent_num = np.zeros([doc_lenth,])
    for docid, seqid, seq in sents:
        sent_num[docid] += 1
    return np.int32(sent_num)



def get_doc_sent(sents, docbow, sent_num, sent_J, pad_id=0):
    index = 0
    D = []
    for id_doc in range(len(sent_num)):
        doc = sents[index : index + sent_num[id_doc]]
        J = int(np.ceil(float(sent_num[id_doc]) / sent_J ))
        for s_id in range(J * sent_J - sent_num[id_doc]):
            doc.append((id_doc, s_id + sent_num[id_doc], [pad_id] * cf.lm_sent_len))  ######## pading 0 of sentence to the doc which is less than 8 sentences
        for id_sent in range(J):
            sent = doc[id_sent * sent_J : (id_sent+1)*sent_J]
            D.append(sent)
        index += sent_num[id_doc]
    return D

def Bow_sents(s,V,tm_ignore):
    s_b = np.zeros(V)
    for w in s:
        s_b[w] += 1
    s_b = np.delete(s_b, list(tm_ignore), axis=0)
    return s_b

def get_batch_tm_lm_all(Doc, docbow, V, tm_ignore, Batch_Size,  pad_id = 0 ):
    x, y, d, m = [], [], [], []
    for idx_sent in range(Batch_Size):
        xx, yy, dd, mm = [], [], [], []
        for docid, seqid, seq in Doc[idx_sent]:
            xx.append(pad(seq[:-1], cf.lm_sent_len, pad_id))
            yy.append(pad(seq[1:], cf.lm_sent_len, pad_id))
            dd.append(docbow[: ,docid] - Bow_sents(seq,V,tm_ignore))
            if seq[0] == pad_id:
                mm.append([0.0] * cf.lm_sent_len)
            else:
                mm.append([1.0] * (len(seq) - 1) + [0.0] * (cf.lm_sent_len - len(seq) + 1))
        x.append(xx)
        y.append(yy)
        d.append(dd)
        m.append(mm)
    return  np.array(x), np.array(y), np.array(d), np.array(m)


