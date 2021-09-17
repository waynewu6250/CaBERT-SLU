"""For model training and inference
Data input should be a single sentence.
"""
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, RMSprop
from transformers import BertTokenizer, BertModel, BertConfig

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import copy
import numpy as np
import collections
from tqdm import tqdm
from collections import Counter, defaultdict

from model import MULTI
from all_data_slot import get_dataloader
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

    if opt.datatype == "mixatis" or opt.datatype == "mixsnips":
        # ATIS Dataset
        X_train, y_train, _ = zip(*train_data)
        X_test, y_test, _ = zip(*test_data)
    elif opt.datatype == "semantic":
        # Semantic parsing Dataset
        X, y = zip(*train_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    elif opt.datatype == "e2e" or opt.datatype == "sgd":
        # Microsoft Dialogue Dataset / SGD Dataset
        all_data = []
        dialogue_id = {}
        dialogue_counter = 0
        counter = 0
        for data in train_data:
            for instance in data:
                all_data.append(instance)
                dialogue_id[counter] = dialogue_counter
                counter += 1
            dialogue_counter += 1
        indices = np.random.permutation(len(all_data))
        train = np.array(all_data)[indices[:int(len(all_data)*0.7)]]#[:10000]
        test = np.array(all_data)[indices[int(len(all_data)*0.7):]]#[:100]
    
    train_loader = get_dataloader(train, len(dic), len(slot_dic), opt)
    val_loader = get_dataloader(test, len(dic), len(slot_dic), opt)
    
    # model
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    
    model = MULTI(opt, len(dic), len(slot_dic))
    if opt.model_path:
        model.load_state_dict(torch.load(opt.model_path))
        print("Pretrained model has been loaded.\n")
    else:
        print("Train from scratch...")
    model = model.to(device)

    # optimizer, criterion
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'gamma', 'beta']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #     'weight_decay_rate': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #     'weight_decay_rate': 0.0}
    # ]
    # optimizer = BertAdam(optimizer_grouped_parameters,lr=opt.learning_rate_bert, warmup=.1)

    optimizer = Adam(model.parameters(), weight_decay=0.01, lr=opt.learning_rate_classifier)
    if opt.data_mode == 'single':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)
    criterion2 = nn.CrossEntropyLoss(reduction='sum').to(device)
    best_loss = 100
    best_accuracy = 0
    best_f1 = 0

    # Start training
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
        for (captions_t, masks, labels, slot_labels) in tqdm(train_loader):

            captions_t = captions_t.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            slot_labels = slot_labels.to(device)
            slot_labels = slot_labels.reshape(-1)

            optimizer.zero_grad()

            encoder_logits, decoder_logits, slot_logits = model(captions_t)
            train_loss = criterion(encoder_logits, labels)
            decoder_logits = decoder_logits.view(-1, len(dic))

            slabels = labels.unsqueeze(1)
            slabels = slabels.repeat(1, opt.maxlen, 1)
            slabels = slabels.view(-1, len(dic))

            train_loss += criterion(decoder_logits, slabels)
            train_loss += criterion2(slot_logits, slot_labels)

            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss
            P, R, F1, acc = f1_score_intents(encoder_logits, labels)
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
        for (captions_t, masks, labels, slot_labels) in val_loader:

            captions_t = captions_t.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            slot_labels = slot_labels.to(device)
            slot_labels = slot_labels.reshape(-1)
            
            with torch.no_grad():
                encoder_logits, decoder_logits, slot_logits = model(captions_t)
            val_loss = criterion(encoder_logits, labels)
            decoder_logits = decoder_logits.view(-1, len(dic))
            slabels = labels.unsqueeze(1)
            slabels = slabels.repeat(1, opt.maxlen, 1)
            slabels = slabels.view(-1, len(dic))
            val_loss += criterion(decoder_logits, slabels)

            total_val_loss += val_loss
            P, R, F1, acc = f1_score_intents(encoder_logits, labels)
            total_P += P
            total_R += R
            total_F1 += F1
            total_acc += acc
            ccounter += 1

            _, index = torch.topk(slot_logits, k=1, dim=-1)
            evaluate_iob(index, slot_labels, slot_dic, stats)

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
        # for label in stats:
        #     if label != 'total':
        #         p, r, f1 = prf(stats[label])
        #         print(f'{label:4s}: P = {p:.4f}, R = {r:.4f}, F1 = {f1:.4f}')
        
        if f1 > best_f1:
            print('saving with loss of {}'.format(total_val_loss),
                  'improved over previous {}'.format(best_loss))
            best_loss = total_val_loss
            best_accuracy = val_acc
            best_f1 = f1
            best_stats = copy.deepcopy(stats)

            torch.save(model.state_dict(), 'checkpoints/best_{}_{}_baseline.pth'.format(opt.datatype, opt.data_mode))
        
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
    print('Dictionary to use: ', opt.dic_path)

    # dataset
    with open(opt.dic_path, 'rb') as f:
        dic = pickle.load(f)
    reverse_dic = {v: k for k,v in dic.items()}
    with open(opt.slot_path, 'rb') as f:
        slot_dic = pickle.load(f)
    with open(opt.train_path, 'rb') as f:
        train_data = pickle.load(f)
    if opt.test_path:
        with open(opt.test_path, 'rb') as f:
            test_data = pickle.load(f)

    if opt.datatype == "atis":
        # ATIS Dataset
        X_train, y_train, _ = zip(*train_data)
        X_test, y_test, _ = zip(*test_data)
    elif opt.datatype == "semantic":
        # Semantic parsing Dataset
        X, y = zip(*train_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    elif opt.datatype == "e2e" or opt.datatype == "sgd":
        # Microsoft Dialogue Dataset / SGD Dataset
        all_data = []
        dialogue_id = {}
        dialogue_counter = 0
        counter = 0
        for data in train_data:
            for instance in data:
                all_data.append(instance)
                dialogue_id[counter] = dialogue_counter
                counter += 1
            dialogue_counter += 1
        indices = np.random.permutation(len(all_data))
        X_train = np.array(all_data)[indices[:int(len(all_data)*0.7)]]#[:10000]
        X_test = np.array(all_data)[indices[int(len(all_data)*0.7):]]#[:100]

    X_train, mask_train = load_data(X_train)
    X_test, mask_test = load_data(X_test)
    
    # model
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    
    model = MULTI(opt, len(dic), len(slot_dic))
    if opt.model_path:
        model.load_state_dict(torch.load(opt.model_path))
        print("Pretrained model has been loaded.\n")
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Store embeddings
    if opt.test_mode == "embedding":
        
        train_loader = get_dataloader(X_train, y_train, mask_train, opt)

        results = collections.defaultdict(list)
        model.eval()
        for i, (captions_t, labels, masks) in enumerate(train_loader):
            
            captions_t = captions_t.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            with torch.no_grad():
                hidden_states, pooled_output, outputs = model(captions_t, masks)
                print("Saving Data: %d" % i)

                for ii in range(len(labels)):
                    key = labels[ii].data.cpu().item()
                    
                    embedding = pooled_output[ii].data.cpu().numpy().reshape(-1)
                    word_embeddings = hidden_states[-1][ii].data.cpu().numpy()
                    
                    tokens = tokenizer.convert_ids_to_tokens(captions_t[ii].data.cpu().numpy())
                    tokens = [token for token in tokens if token != "[CLS]" and token != "[SEP]" and token != "[PAD]"]
                    original_sentence = " ".join(tokens)
                    results[key].append((original_sentence, embedding, word_embeddings))

        torch.save(results, embedding_path)
    
    # Run test classification
    elif opt.test_mode == "data":
        
        # Single instance
        # index = np.random.randint(0, len(X_test), 1)[0]
        # input_ids = X_test[index]
        # attention_masks = mask_test[index]
        # print(" ".join(tokenizer.convert_ids_to_tokens(input_ids)))

        # captions_t = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        # mask = torch.LongTensor(attention_masks).unsqueeze(0).to(device)
        # with torch.no_grad():
        #     pooled_output, outputs = model(captions_t, mask)
        # print("Predicted label: ", reverse_dic[torch.max(outputs, 1)[1].item()])
        # print("Real label: ", reverse_dic[y_test[index]])

        # Validation Phase
        test_loader = get_dataloader(X_test, y_test, mask_test, len(dic), opt)
        
        error_ids = []
        pred_labels = []
        real_labels = []
        test_corrects = 0
        totals = 0
        model.eval()
        for i, (captions_t, labels, masks) in enumerate(test_loader):
            print('predict batches: ', i)

            captions_t = captions_t.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            with torch.no_grad():
                _, pooled_output, outputs = model(captions_t, masks)
                co, to = calc_score(outputs, labels)
                test_corrects += co
                totals += to

                if opt.data_mode == 'single':
                    idx = torch.max(outputs, 1)[1] != labels
                    wrong_ids = [tokenizer.convert_ids_to_tokens(caption, skip_special_tokens=True) for caption in captions_t[idx]]
                    error_ids += wrong_ids
                    pred_labels += [reverse_dic[label.item()] for label in torch.max(outputs, 1)[1][idx]]
                    real_labels += [reverse_dic[label.item()] for label in labels[idx]]
                else:
                    for i, logits in enumerate(outputs):
                        log = torch.sigmoid(logits)
                        correct = (labels[i][torch.where(log>0.5)[0]]).sum()
                        total = len(torch.where(labels[i]==1)[0])
                        if correct != total:
                            wrong_caption = tokenizer.convert_ids_to_tokens(captions_t[i], skip_special_tokens=True)
                            error_ids.append(wrong_caption)
                            pred_ls = [reverse_dic[p] for p in torch.where(log>0.5)[0].detach().cpu().numpy()]
                            real_ls = [reverse_dic[i] for i, r in enumerate(labels[i].detach().cpu().numpy()) if r == 1]
                            pred_labels.append(pred_ls)
                            real_labels.append(real_ls)

        with open('error_analysis/{}_{}.txt'.format(opt.datatype, opt.data_mode), 'w') as f:
            f.write('----------- Wrong Examples ------------\n')
            for i, (caption, pred, real) in enumerate(zip(error_ids, pred_labels, real_labels)):
                f.write(str(i)+'\n')
                f.write(' '.join(caption)+'\n')
                f.write('Predicted label: {}\n'.format(pred))
                f.write('Real label: {}\n'.format(real))
                f.write('------\n')
        test_acc = test_corrects.double() / test_loader.dataset.num_data if opt.data_mode == 'single' else test_corrects.double() / totals
        print('Test accuracy: {:.4f}'.format(test_acc))

    
    # User defined
    elif opt.test_mode == "user":
        while True:
            print("Please input the sentence: ")
            text = input()
            print("\n======== Predicted Results ========")
            print(text)
            text = "[CLS] " + text + " [SEP]"
            tokenized_text = tokenizer.tokenize(text)
            tokenized_ids = np.array(tokenizer.convert_tokens_to_ids(tokenized_text))[np.newaxis,:]
            
            input_ids = pad_sequences(tokenized_ids, maxlen=opt.maxlen, dtype="long", truncating="post", padding="post").squeeze(0)
            attention_masks = [float(i>0) for i in input_ids]

            captions_t = torch.LongTensor(input_ids).unsqueeze(0).to(device)
            mask = torch.LongTensor(attention_masks).unsqueeze(0).to(device)
            with torch.no_grad():
                pooled_output, outputs = model(captions_t, mask)
            print("Predicted label: ", reverse_dic[torch.max(outputs, 1)[1].item()])
            print("=================================")    
    
    





if __name__ == '__main__':
    import fire
    fire.Fire()
    


            








        








    


    