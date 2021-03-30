import torch as t
from torch.utils.data import Dataset, DataLoader
import pickle
from config import opt
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import numpy as np

class CoreDataset(Dataset):

    def __init__(self, data, num_labels, num_slot_labels, opt):
        self.data = data
        self.num_data = len(self.data)
        self.maxlen = opt.maxlen
        self.num_labels = num_labels
        self.num_slot_labels = num_slot_labels
        self.opt = opt
        
        caps, slots, labels = zip(*self.data)
        self.caps, self.masks = self.load_data(caps, self.maxlen)
        self.slot_labels, _ = self.load_data(slots, self.maxlen)
        self.labels = labels
    
    def load_data(self, X, maxlen):

        input_ids = pad_sequences(X, maxlen=maxlen, dtype="long", truncating="post", padding="post")
        
        attention_masks = []
        for seq in input_ids:
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask)
        return t.tensor(input_ids), t.tensor(attention_masks)

    def __getitem__(self, index):
        
        # caps
        caps = self.caps[index]
        slot_labels = self.slot_labels[index]
        masks = self.masks[index]

        # labels
        label = t.LongTensor(np.array(self.labels[index]))
        labels = t.zeros(self.num_labels).scatter_(0, label, 1)

        return caps, masks, labels, slot_labels
        

    def __len__(self):
        return len(self.data)

def get_dataloader(data, num_labels, num_slot_labels, opt):
    dataset = CoreDataset(data, num_labels, num_slot_labels, opt)
    batch_size = opt.batch_size
    return DataLoader(dataset, 
                      batch_size=batch_size, 
                      shuffle=False)

    
    