import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, RMSprop
from transformers import BertTokenizer, BertModel, BertConfig, AdamW

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import copy
import numpy as np
import collections
from tqdm import tqdm
from more_itertools import collapse
from collections import defaultdict

from model import BertContextNLU
from all_data_context import get_dataloader_context
from config import opt

def load_data(X, maxlen):

    input_ids = pad_sequences(X, maxlen=maxlen, dtype="long", truncating="post", padding="post")
    
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return (input_ids, attention_masks)

def f1_score_intents(outputs, labels):
    
    P, R, F1, acc = 0, 0, 0, 0
    outputs = torch.sigmoid(outputs)

    for i in range(outputs.shape[0]):
        TP, FP, FN = 0, 0, 0
        for j in range(outputs.shape[1]):
            if outputs[i][j] > 0.5 and labels[i][j] == 1:
                TP += 1
            elif outputs[i][j] <= 0.5 and labels[i][j] == 1:
                FN += 1
            elif outputs[i][j] > 0.5 and labels[i][j] == 0:
                FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 += 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        P += precision
        R += recall

        p = (torch.where(outputs[i]>0.5)[0])
        r = (torch.where(labels[i]==1)[0])
        if len(p) == len(r) and (p == r).all():
            acc += 1
        
    P /= outputs.shape[0]
    R /= outputs.shape[0]
    F1 /= outputs.shape[0]
    return P, R, F1, acc


############################################3

def to_spans(l_ids, voc):
    """Convert a list of BIO labels, coded as integers, into spans identified by a beginning, an end, and a label. 
       To allow easy comparison later, we store them in a dictionary indexed by the start position.
    @param l_ids: a list of predicted label indices
    @param voc: label vocabulary dictionary: index to label ex. 0: B-C
    """
    spans = {}
    current_lbl = None
    current_start = None
    for i, l_id in enumerate(l_ids):
        l = voc[l_id]

        if l[0] == 'B': 
            # Beginning of a named entity: B-something.
            if current_lbl:
                # If we're working on an entity, close it.
                spans[current_start] = (current_lbl, i)
            # Create a new entity that starts here.
            current_lbl = l[2:]
            current_start = i
        elif l[0] == 'I':
            # Continuation of an entity: I-something.
            if current_lbl:
                # If we have an open entity, but its label does not
                # correspond to the predicted I-tag, then we close
                # the open entity and create a new one.
                if current_lbl != l[2:]:
                    spans[current_start] = (current_lbl, i)
                    current_lbl = l[2:]
                    current_start = i
            else:
                # If we don't have an open entity but predict an I tag,
                # we create a new entity starting here even though we're
                # not following the format strictly.
                current_lbl = l[2:]
                current_start = i
        else:
            # Outside: O.
            if current_lbl:
                # If we have an open entity, we close it.
                spans[current_start] = (current_lbl, i)
                current_lbl = None
                current_start = None
        if current_lbl != None:
            spans[current_start] = (current_lbl, i+1)
    return spans


def compare(gold, pred, stats, mode='strict'):
    """Compares two sets of spans and records the results for future aggregation.
    @param gold: ground truth
    @param pred: predictions
    @param stats: the final dictionary with keys of different counts including total and specific labels
                  ex. {'total': {'gold': 5, 'pred': 5},
                       'Cause': {'gold': 5, 'pred': 5}}
    """
    for start, (lbl, end) in gold.items():
        stats['total']['gold'] += 1
        stats[lbl]['gold'] += 1
    for start, (lbl, end) in pred.items():
        stats['total']['pred'] += 1
        stats[lbl]['pred'] += 1
    
    if mode == 'strict':
        for start, (glbl, gend) in gold.items():
            if start in pred:
                plbl, pend = pred[start]
                if glbl == plbl and gend == pend:
                    stats['total']['corr'] += 1
                    stats[glbl]['corr'] += 1
    
    elif mode == 'partial':
        for gstart, (glbl, gend) in gold.items():
            for pstart, (plbl, pend) in pred.items():
                if glbl == plbl:
                    g = set(range(gstart, gend+1))
                    p = set(range(pstart, pend+1))
                    if len(g & p) / max(len(g), len(p)) >= opt.token_percent:
                        stats['total']['corr'] += 1
                        stats[glbl]['corr'] += 1
                        break


def evaluate_iob(predicted, gold, label_field, stats):
    """This function will evaluate the model from bert dataloader pipeline.
    """
    gold_cpu = gold.cpu().numpy()
    pred_cpu = predicted.cpu().numpy()
    gold_cpu = list(gold_cpu.reshape(-1))
    pred_cpu = list(pred_cpu.reshape(-1))
    # pred_cpu = [l for sen in predicted for l in sen]

    id2label = {v:k for k,v in label_field.items()}
    # Compute spans for the gold standard and prediction.
    gold_spans = to_spans(gold_cpu, id2label)
    pred_spans = to_spans(pred_cpu, id2label)

    # Finally, update the counts for correct, predicted and gold-standard spans.
    compare(gold_spans, pred_spans, stats, 'strict')

def prf(stats):
    """
    Computes precision, recall and F-score, given a dictionary that contains
    the counts of correct, predicted and gold-standard items.
    @params stats: the final statistics
    """
    if stats['pred'] == 0:
        return 0, 0, 0
    p = stats['corr']/stats['pred']
    r = stats['corr']/stats['gold']
    if p > 0 and r > 0:
        f = 2*p*r/(p+r)
    else:
        f = 0
    return p, r, f