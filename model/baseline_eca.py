import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel

class ECA(nn.Module):
    
    def __init__(self, opt, num_labels=2, num_slot_labels=10):
        super(ECA, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.embedding = nn.Embedding(len(self.tokenizer.vocab), 256)

        self.utterance_encoder = nn.LSTM(input_size=256, 
                           hidden_size=256,
                           bidirectional=True,
                           batch_first=True, 
                           num_layers=1)
        self.conversation_layer = nn.LSTM(input_size=512, 
                           hidden_size=256,
                           bidirectional=True,
                           batch_first=True, 
                           num_layers=1)
        self.dense1 = nn.Linear(512, 256)
        self.dense2 = nn.Linear(256, num_labels)
        
        self.slot_decoder = AttnDecoderRNN(256, opt)

        self.classifier_slot = nn.Linear(512, num_slot_labels)
        nn.init.xavier_normal_(self.classifier_slot.weight)
        #self.dropout = nn.Dropout(0.1)
        self.num_labels = num_labels
        self.num_slot_labels = num_slot_labels

        self.dropout = nn.Dropout(0.1)
        
        self.opt = opt

    def forward(self, result_ids, result_token_masks, result_masks, lengths, result_slot_labels, labels, y_caps, y_masks):
        
        # Utterance Encoder
        b,d,t = result_ids.shape
        result_ids = result_ids.view(-1, t)
        X = self.embedding(result_ids)
        rnn_out, encoder_hidden = self.utterance_encoder(X)

        # pooling & conversation
        pooled = rnn_out[:,-1,:].view(b,d,2*256)
        out, hidden = self.conversation_layer(pooled)
        out = self.dense1(out)
        logits = self.dense2(out)

        # Remove padding
        logits_no_pad = []
        labels_no_pad = []
        for i in range(b):
            logits_no_pad.append(logits[i,:lengths[i],:])
            labels_no_pad.append(labels[i,:lengths[i],:])
        logits = torch.cat(logits_no_pad, dim=0)
        labels = torch.cat(labels_no_pad, dim=0)

        # Slot Decoder
        decoder_hidden = encoder_hidden
        slot_outputs = torch.zeros(*rnn_out.shape, device=self.device)
        
        for di in range(t):
            decoder_output, decoder_hidden = self.slot_decoder(decoder_hidden, rnn_out, di)
            slot_outputs[:,di,:] = decoder_output.squeeze(1)
        #decoder_outputs = self.dropout(decoder_outputs)
        slot_outputs = self.dropout(slot_outputs)
        slot_logits = self.classifier_slot(slot_outputs)
        slot_logits = slot_logits.view(-1, self.num_slot_labels)

        return logits, labels, slot_logits



class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, opt):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = 256
        self.max_length = opt.maxlen

        self.attn = nn.Linear(self.hidden_size * 4, 1)
        self.attn_combine = nn.Linear(self.hidden_size * 4, self.hidden_size)
        self.rnn_token = nn.LSTM(input_size=self.hidden_size, 
                           hidden_size=self.hidden_size,
                           bidirectional=True,
                           batch_first=True, 
                           num_layers=1)
        
        self.W = nn.Parameter(torch.zeros(self.hidden_size*2,1))
        self.v = nn.Parameter(torch.zeros(1))

    def forward(self, hidden, encoder_outputs, di):
        
        b, t, h = encoder_outputs.shape

        # repeat decoder hidden
        decoder_hidden = hidden[0].view(-1, 2*self.hidden_size) # (b,2h)
        hidden_repeat = decoder_hidden.unsqueeze(1) # (b,1,2h)
        hidden_repeat = hidden_repeat.repeat(1,t,1) # (b,t,2h)

        # attention
        attn_weights = self.attn(torch.cat((encoder_outputs, hidden_repeat), 2)) # (b,t,1)
        attn_weights = F.softmax(attn_weights, dim=1) # (b,t,1)
        attn_applied = torch.bmm(encoder_outputs.transpose(2,1), attn_weights).squeeze(2) # (b,2h)

        output = torch.cat((encoder_outputs[:,di,:], attn_applied), dim=1) # (b,4h)

        # linear layer
        output = self.attn_combine(output) # (b,h)
        output = F.relu(output)
        output = output.unsqueeze(1) # (b,1,h)
        output, hidden = self.rnn_token(output, hidden)

        return output, hidden
     