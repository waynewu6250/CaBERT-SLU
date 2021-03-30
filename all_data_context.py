import torch as t
from torch.utils.data import Dataset, DataLoader
import pickle
from config import opt
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing.sequence import pad_sequences

class Turns:
    def __init__(self, token_ids, slot_ids):
        token_ids, mask = self.load_data(token_ids)
        slot_ids, _ = self.load_data(slot_ids)
        self.token_ids = np.stack(token_ids, axis=0)
        self.slot_ids = np.stack(slot_ids, axis=0)
        self.attention_masks = mask

    def load_data(self, X):
        input_ids = pad_sequences(X, maxlen=60, dtype="long", truncating="post", padding="post")
        
        attention_masks = []
        for seq in input_ids:
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask)
        
        return input_ids, attention_masks

class CoreDataset(Dataset):

    def __init__(self, data, dic, slot_dic, opt):
        self.data = data
        self.dic = dic
        self.slot_dic = slot_dic
        self.opt = opt
        self.num_labels = len(dic)
        self.num_slot_labels = len(slot_dic)

        self.X_turns, self.Y_turns = self.postprocess()
        self.num_data = sum([len(turn.token_ids) for turn in self.X_turns])
    
    def postprocess(self):
        
        dialogs = []
        y_slots = []
        y_labels = []
        
        for dialog in self.data:
            utts, slots, labels = zip(*dialog)
            dialogs.append(utts)
            y_slots.append(slots)
            y_labels.append(labels)

        X_turns = np.array([Turns(turns, slots) for turns, slots in zip(dialogs, y_slots)])
        Y_turns = np.array(y_labels)
        
        return X_turns, Y_turns

    def __getitem__(self, index):

        # onehot
        labels = self.Y_turns[index]
        new_labels = t.zeros((len(labels), self.num_labels)).long()
        for i, label in enumerate(labels):
            label = t.LongTensor(np.array(label))
            label = t.zeros(self.num_labels).scatter_(0, label, 1)
            new_labels[i] = label

        return self.X_turns[index], new_labels
        
    def __len__(self):
        return len(self.X_turns)

def collate_fn(batch):

    X_turns, Y_update = zip(*batch)
    num_labels = Y_update[0].shape[1]

    lengths = [i.token_ids.shape[0] for i in X_turns]
    lengths = t.LongTensor(lengths)
    
    max_len = max([i.token_ids.shape[0] for i in X_turns])
    max_dim = max([i.token_ids.shape[1] for i in X_turns])
    result_ids = t.zeros((len(X_turns), max_len, max_dim)).long()
    result_token_masks = t.zeros((len(X_turns), max_len, max_dim)).long()
    result_masks = t.zeros((len(X_turns), max_len)).long()
    result_slot_labels = t.zeros((len(X_turns), max_len, max_dim)).long()
    result_labels = t.ones((len(X_turns), max_len, num_labels))*-1

    for i in range(len(X_turns)):
        len1 = X_turns[i].token_ids.shape[0]
        dim1 = X_turns[i].token_ids.shape[1]
        result_ids[i, :len1, :dim1] = t.Tensor(X_turns[i].token_ids)
        result_token_masks[i, :len1, :dim1] = t.Tensor(X_turns[i].attention_masks)
        for j in range(lengths[i]):
            result_masks[i][j] = 1
        result_slot_labels[i, :len1, :dim1] = t.Tensor(X_turns[i].slot_ids)
        result_labels[i, :len1, :] = Y_update[i]

    return result_ids, result_token_masks, result_masks, lengths, result_slot_labels, result_labels

def get_dataloader_context(data, dic, slot_dic, opt):
    dataset = CoreDataset(data, dic, slot_dic, opt)
    batch_size = opt.batch_size
    return DataLoader(dataset, 
                      batch_size=batch_size, 
                      shuffle=False,
                      collate_fn= lambda x: collate_fn(x))

######################################################################

if __name__ == '__main__':
    
    with open(opt.dic_path_with_tokens, 'rb') as f:
        dic = pickle.load(f)
    with open(opt.slot_path, 'rb') as f:
        slot_dic = pickle.load(f)
    with open(opt.train_path, 'rb') as f:
        train_data = pickle.load(f)
    np.random.seed(0)

    indices = np.arange(len(train_data)) #np.random.permutation(len(train_data))
    train = np.array(train_data)[indices[:int(len(train_data)*0.7)]]
    test = np.array(train_data)[indices[int(len(train_data)*0.7):]]

    train_loader = get_dataloader_context(train, dic, slot_dic, opt)

    for result_ids, result_token_masks, result_masks, lengths, result_slot_labels, result_labels in train_loader:
        print(result_ids[0])
        print(result_token_masks[0])
        print(result_masks[0])
        print(lengths[0])
        print(result_slot_labels[0])
        print(result_labels[0])
        dae
    