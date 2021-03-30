import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel

class MULTI(nn.Module):
    
    def __init__(self, opt, num_labels=2, num_slot_labels=10):
        super(MULTI, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.embedding = nn.Embedding(len(self.tokenizer.vocab), 64)
        self.rnn_sentence = nn.LSTM(input_size=64, 
                           hidden_size=64,
                           bidirectional=True,
                           batch_first=True, 
                           num_layers=1)
        self.decoder = AttnDecoderRNN(64, opt)
        self.slot_decoder = AttnDecoderRNN(64, opt)

        self.classifier1 = nn.Linear(128, num_labels)
        nn.init.xavier_normal_(self.classifier1.weight)
        self.classifier2 = nn.Linear(128, num_labels)
        nn.init.xavier_normal_(self.classifier2.weight)
        self.classifier_slot = nn.Linear(128, num_slot_labels)
        nn.init.xavier_normal_(self.classifier_slot.weight)
        #self.dropout = nn.Dropout(0.1)
        self.num_labels = num_labels
        self.num_slot_labels = num_slot_labels
        
        self.opt = opt

    def forward(self, x_inputs):
        
        # Encoder
        X = self.embedding(x_inputs)
        rnn_out, encoder_hidden = self.rnn_sentence(X)
        #rnn_out = self.dropout(rnn_out)
        logits = self.classifier1(rnn_out[:,-1,:])
        encoder_logits = logits

        # Decoder
        decoder_hidden = encoder_hidden
        decoder_outputs = torch.zeros(*rnn_out.shape, device=self.device)
        
        for di in range(x_inputs.shape[1]):
            decoder_output, decoder_hidden = self.decoder(decoder_hidden, rnn_out, di)
            decoder_outputs[:,di,:] = decoder_output.squeeze(1)
        #decoder_outputs = self.dropout(decoder_outputs)
        decoder_logits = self.classifier2(decoder_outputs)

        # Slot Decoder
        decoder_hidden = encoder_hidden
        slot_outputs = torch.zeros(*rnn_out.shape, device=self.device)
        
        for di in range(x_inputs.shape[1]):
            decoder_output, decoder_hidden = self.slot_decoder(decoder_hidden, rnn_out, di)
            slot_outputs[:,di,:] = decoder_output.squeeze(1)
        #decoder_outputs = self.dropout(decoder_outputs)
        slot_logits = self.classifier_slot(slot_outputs)
        slot_logits = slot_logits.view(-1, self.num_slot_labels)

        return encoder_logits, decoder_logits, slot_logits

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, opt):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = 64
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
        decoder_hidden = hidden[0].view(-1, 128) # (b,2h)
        hidden_repeat = decoder_hidden.unsqueeze(1) # (b,1,2h)
        hidden_repeat = hidden_repeat.repeat(1,t,1) # (b,t,2h)

        # attention
        attn_weights = self.attn(torch.cat((encoder_outputs, hidden_repeat), 2)) # (b,t,1)
        attn_weights = F.softmax(attn_weights, dim=1) # (b,t,1)
        attn_applied = torch.bmm(encoder_outputs.transpose(2,1), attn_weights).squeeze(2) # (b,2h)

        # # slot-gated:
        # print(attn_applied.shape)
        # print(encoder_outputs[:,-1,:].shape)
        # print(self.W.shape)
        # print(self.v.shape)
        # g = torch.sum(self.v * torch.tanh(attn_applied + encoder_outputs[:,-1,:] * self.W), dim=1) # (b,)
        # g = g.expand(dim=1).repeat(1,1,2*h) # (b,1)
        output = torch.cat((encoder_outputs[:,di,:], attn_applied), dim=1) # (b,4h)

        # linear layer
        output = self.attn_combine(output) # (b,h)
        output = F.relu(output)
        output = output.unsqueeze(1) # (b,1,h)
        output, hidden = self.rnn_token(output, hidden)

        return output, hidden
     