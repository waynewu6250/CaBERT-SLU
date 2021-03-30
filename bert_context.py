"""For model training and inference (multi dialogue act & slot detection)
"""
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
from collections import defaultdict, Counter

from model import BertContextNLU, ECA
from all_data_context import get_dataloader_context
from config import opt
from utils import *

def train(**kwargs):
    
    # attributes
    for k, v in kwargs.items():
        setattr(opt, k, v)
    np.random.seed(0)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False

    print('Dataset to use: ', opt.train_path)
    print('Dictionary to use: ', opt.dic_path_with_tokens)
    print('Data Type: ', opt.datatype)
    print('Use pretrained weights: ', opt.retrain)

    # dataset
    with open(opt.dic_path_with_tokens, 'rb') as f:
        dic = pickle.load(f)
    with open(opt.slot_path, 'rb') as f:
        slot_dic = pickle.load(f)
    with open(opt.train_path, 'rb') as f:
        train_data = pickle.load(f)

    # Microsoft Dialogue Dataset / SGD Dataset
    indices = np.random.permutation(len(train_data))
    train = np.array(train_data)[indices[:int(len(train_data)*0.7)]]#[:1000]
    test = np.array(train_data)[indices[int(len(train_data)*0.7):]]#[:100]
    
    train_loader = get_dataloader_context(train, dic, slot_dic, opt)
    val_loader = get_dataloader_context(test, dic, slot_dic, opt)

    # label tokens
    intent_tokens = [intent for name, (tag, intent) in dic.items()]
    intent_tok, mask_tok = load_data(intent_tokens, 10)
    intent_tokens = torch.zeros(len(intent_tok), 10).long().to(device)
    mask_tokens = torch.zeros(len(mask_tok), 10).long().to(device)
    for i in range(len(intent_tok)):
        intent_tokens[i] = torch.tensor(intent_tok[i])
    for i in range(len(mask_tok)):
        mask_tokens[i] = torch.tensor(mask_tok[i])
    
    # model
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    
    model = BertContextNLU(config, opt, len(dic), len(slot_dic))
    # model = ECA(opt, len(dic), len(slot_dic))
    
    if opt.model_path:
        model.load_state_dict(torch.load(opt.model_path))
        print("Pretrained model has been loaded.\n")
    else:
        print("Train from scratch...")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), weight_decay=0.01, lr=opt.learning_rate_bert)
    criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)
    criterion2 = nn.CrossEntropyLoss(reduction='sum').to(device)

    best_loss = 100
    best_accuracy = 0
    best_f1 = 0

    #################################### Start training ####################################
    for epoch in range(opt.epochs):
        print("====== epoch %d / %d: ======"% (epoch+1, opt.epochs))

        # Training Phase
        total_train_loss = 0
        total_P = 0
        total_R = 0
        total_F1 = 0
        total_acc = 0
        model.train()
        ccounter = 0
        for (result_ids, result_token_masks, result_masks, lengths, result_slot_labels, result_labels) in tqdm(train_loader):

            result_ids = result_ids.to(device)
            result_token_masks = result_token_masks.to(device)
            result_masks = result_masks.to(device)
            lengths = lengths.to(device)
            result_slot_labels = result_slot_labels.to(device)
            result_slot_labels = result_slot_labels.reshape(-1)
            result_labels = result_labels.to(device)

            optimizer.zero_grad()

            outputs, labels, slot_out = model(result_ids, result_token_masks, result_masks, lengths, result_slot_labels, result_labels, intent_tokens, mask_tokens)
            train_loss = criterion(outputs, labels)
            slot_loss = criterion2(slot_out, result_slot_labels)
            total_loss = train_loss + slot_loss
            
            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss
            P, R, F1, acc = f1_score_intents(outputs, labels)
            total_P += P
            total_R += R
            total_F1 += F1
            total_acc += acc
            ccounter += 1

        print('Average train loss: {:.4f} '.format(total_train_loss / train_loader.dataset.num_data))
        precision = total_P / ccounter
        recall = total_R / ccounter
        f1 = total_F1 / ccounter
        print(f'P = {precision:.4f}, R = {recall:.4f}, F1 = {f1:.4f}')
        print('Accuracy: ', total_acc/train_loader.dataset.num_data)
        

        # Validation Phase
        total_val_loss = 0
        total_P = 0
        total_R = 0
        total_F1 = 0
        total_acc = 0
        model.eval()
        ccounter = 0
        stats = defaultdict(Counter)
        for (result_ids, result_token_masks, result_masks, lengths, result_slot_labels, result_labels) in val_loader:

            result_ids = result_ids.to(device)
            result_token_masks = result_token_masks.to(device)
            result_masks = result_masks.to(device)
            lengths = lengths.to(device)
            result_slot_labels = result_slot_labels.to(device)
            result_slot_labels = result_slot_labels.reshape(-1)
            result_labels = result_labels.to(device)
            
            with torch.no_grad():
                outputs, labels, predicted_slot_outputs  = model(result_ids, result_token_masks, result_masks, lengths, result_slot_labels, result_labels, intent_tokens, mask_tokens)
            val_loss = criterion(outputs, labels)

            total_val_loss += val_loss
            P, R, F1, acc = f1_score_intents(outputs, labels)
            total_P += P
            total_R += R
            total_F1 += F1
            total_acc += acc
            ccounter += 1

            _, index = torch.topk(predicted_slot_outputs, k=1, dim=-1)
            evaluate_iob(index, result_slot_labels, slot_dic, stats)

        print('========= Validation =========')
        print('Average val loss: {:.4f} '.format(total_val_loss / val_loader.dataset.num_data))

        precision = total_P / ccounter
        recall = total_R / ccounter
        f1 = total_F1 / ccounter
        print(f'P = {precision:.4f}, R = {recall:.4f}, F1 = {f1:.4f}')
        print('Accuracy: ', total_acc/val_loader.dataset.num_data)
        val_acc = total_acc/val_loader.dataset.num_data

        # print slot stats
        p_slot, r_slot, f1_slot = prf(stats['total'])
        print('========= Slot =========')
        print(f'Slot Score: P = {p_slot:.4f}, R = {r_slot:.4f}, F1 = {f1_slot:.4f}')
        
        if f1 > best_f1:
            print('saving with loss of {}'.format(total_val_loss),
                  'improved over previous {}'.format(best_loss))
            best_loss = total_val_loss
            best_accuracy = val_acc
            best_f1 = f1
            best_stats = copy.deepcopy(stats)

            torch.save(model.state_dict(), 'checkpoints/best_{}_{}.pth'.format(opt.datatype, opt.data_mode))
        
        print()
    print('Best total val loss: {:.4f}'.format(total_val_loss))
    print('Best Test Accuracy: {:.4f}'.format(best_accuracy))
    print('Best F1 Score: {:.4f}'.format(best_f1))

    p_slot, r_slot, f1_slot = prf(best_stats['total'])
    print('Final evaluation on slot filling of the validation set:')
    print(f'Overall: P = {p_slot:.4f}, R = {r_slot:.4f}, F1 = {f1_slot:.4f}')


#####################################################################


def test(**kwargs):

    # attributes
    for k, v in kwargs.items():
        setattr(opt, k, v)
    np.random.seed(0)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False

    print('Dataset to use: ', opt.train_path)
    print('Dictionary to use: ', opt.dic_path_with_tokens)

    # dataset
    with open(opt.dic_path_with_tokens, 'rb') as f:
        dic = pickle.load(f)
    print(dic)
    with open(opt.slot_path, 'rb') as f:
        slot_dic = pickle.load(f)
    reverse_dic = {v[0]: k for k,v in dic.items()}
    with open(opt.train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(opt.test_path, 'rb') as f:
        test_data = pickle.load(f)

    
    # Microsoft Dialogue Dataset / SGD Dataset
    indices = np.random.permutation(len(train_data))
    train = np.array(train_data)[indices[:int(len(train_data)*0.7)]]
    test = np.array(train_data)[indices[int(len(train_data)*0.7):]][:1000]

    train_loader = get_dataloader_context(train, dic, slot_dic, opt)
    test_loader = get_dataloader_context(test, dic, slot_dic, opt)

    # label tokens
    intent_tokens = [intent for name, (tag, intent) in dic.items()]
    intent_tok, mask_tok = load_data(intent_tokens, 10)
    intent_tokens = torch.zeros(len(intent_tok), 10).long().to(device)
    mask_tokens = torch.zeros(len(mask_tok), 10).long().to(device)
    for i in range(len(intent_tok)):
        intent_tokens[i] = torch.tensor(intent_tok[i])
    for i in range(len(mask_tok)):
        mask_tokens[i] = torch.tensor(mask_tok[i])
    
    # model
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    
    model = BertContextNLU(config, opt, len(dic), len(slot_dic))

    if opt.model_path:
        model.load_state_dict(torch.load(opt.model_path))
        print("Pretrained model {} has been loaded.".format(opt.model_path))
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    # Run multi-intent validation
    if opt.test_mode == "validation":
        
        total_P = 0
        total_R = 0
        total_F1 = 0
        total_acc = 0
        model.eval()
        ccounter = 0
        stats = defaultdict(Counter)
        for (result_ids, result_token_masks, result_masks, lengths, result_slot_labels, result_labels) in tqdm(test_loader):

            result_ids = result_ids.to(device)
            result_token_masks = result_token_masks.to(device)
            result_masks = result_masks.to(device)
            lengths = lengths.to(device)
            result_slot_labels = result_slot_labels.to(device)
            result_slot_labels = result_slot_labels.reshape(-1)
            result_labels = result_labels.to(device)
            
            with torch.no_grad():
                outputs, labels, predicted_slot_outputs  = model(result_ids, result_token_masks, result_masks, lengths, result_slot_labels, result_labels, intent_tokens, mask_tokens)

            P, R, F1, acc = f1_score_intents(outputs, labels)
            total_P += P
            total_R += R
            total_F1 += F1
            total_acc += acc
            ccounter += 1

            _, index = torch.topk(predicted_slot_outputs, k=1, dim=-1)
            evaluate_iob(index, result_slot_labels, slot_dic, stats)

        precision = total_P / ccounter
        recall = total_R / ccounter
        f1 = total_F1 / ccounter
        print(f'P = {precision:.4f}, R = {recall:.4f}, F1 = {f1:.4f}')
        print('Accuracy: ', total_acc/test_loader.dataset.num_data)

        # print slot stats
        p_slot, r_slot, f1_slot = prf(stats['total'])
        print('========= Slot =========')
        print(f'Slot Score: P = {p_slot:.4f}, R = {r_slot:.4f}, F1 = {f1_slot:.4f}')
    
    # Run test classification
    elif opt.test_mode == "data":

        # Validation Phase
        pred_labels = []
        real_labels = []
        error_ids = []
        total_P, total_R, total_F1, total_acc = 0, 0, 0, 0
        ccounter = 0
        stats = defaultdict(Counter)
        model.eval()
        print(len(test_loader.dataset))
        for num, (result_ids, result_token_masks, result_masks, lengths, result_slot_labels, result_labels) in enumerate(test_loader):
            print('predict batches: ', num)

            result_ids = result_ids.to(device)
            result_token_masks = result_token_masks.to(device)
            result_masks = result_masks.to(device)
            lengths = lengths.to(device)
            result_slot_labels = result_slot_labels.to(device)
            result_slot_labels = result_slot_labels.reshape(-1)
            result_labels = result_labels.to(device)

            # Remove padding
            texts_no_pad = []
            for i in range(len(result_ids)):
                texts_no_pad.append(result_ids[i,:lengths[i],:])
            texts_no_pad = torch.vstack(texts_no_pad)
            
            with torch.no_grad():
                outputs, labels, predicted_slot_outputs, ffscores  = model(result_ids, result_token_masks, result_masks, lengths, result_slot_labels, result_labels, intent_tokens, mask_tokens)

                # total
                P, R, F1, acc = f1_score_intents(outputs, labels)
                total_P += P
                total_R += R
                total_F1 += F1
                total_acc += acc
                
                ccounter += 1

                _, index = torch.topk(predicted_slot_outputs, k=1, dim=-1)
                evaluate_iob(index, result_slot_labels, slot_dic, stats)

                for i, logits in enumerate(outputs):
                    log = torch.sigmoid(logits)
                    correct = (labels[i][torch.where(log>0.5)[0]]).sum()
                    total = len(torch.where(labels[i]==1)[0])
                    wrong_caption = tokenizer.convert_ids_to_tokens(texts_no_pad[i], skip_special_tokens=True)
                    error_ids.append(wrong_caption)
                    pred_ls = [p for p in torch.where(log>0.5)[0].detach().cpu().numpy()]
                    real_ls = [i for i, r in enumerate(labels[i].detach().cpu().numpy()) if r == 1]
                    pred_labels.append(pred_ls)
                    real_labels.append(real_ls)

        with open('error_analysis/{}_{}_context_slots.txt'.format(opt.datatype, opt.data_mode), 'w') as f:
            f.write('----------- Examples ------------\n')
            for i, (caption, pred, real) in enumerate(zip(error_ids, pred_labels, real_labels)):
                f.write(str(i)+'\n')
                f.write(' '.join(caption)+'\n')
                p_r = [reverse_dic[p] for p in pred]
                r_r = [reverse_dic[r] for r in real]
                f.write('Predicted label: {}\n'.format(p_r))
                f.write('Real label: {}\n'.format(r_r))
                f.write('------\n')
        precision = total_P / ccounter
        recall = total_R / ccounter
        f1 = total_F1 / ccounter
        print(f'P = {precision:.4f}, R = {recall:.4f}, F1 = {f1:.4f}')
        print('Accuracy: ', total_acc/test_loader.dataset.num_data)

        print(len(ffscores))
        with open('ffscores.pkl', 'wb') as f:
            pickle.dump(ffscores, f)



if __name__ == '__main__':
    import fire
    fire.Fire()
    


            








        








    


    