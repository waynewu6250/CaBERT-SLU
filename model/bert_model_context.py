import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel
from model.transformer import TransformerModel
from model.transformer_new import Transformer
from model.CHAN import ContextAttention
from model.torchcrf import CRF
from model.mia import MutualIterativeAttention

class BertContextNLU(nn.Module):
    
    def __init__(self, config, opt, num_labels=2, num_slot_labels=144):
        super(BertContextNLU, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_labels = num_labels
        self.num_slot_labels = num_slot_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
        self.dropout = nn.Dropout(0.1)
        self.hidden_size = config.hidden_size
        self.rnn_hidden = opt.rnn_hidden

        #########################################

        # transformer
        self.transformer_model = TransformerModel(ninp=self.hidden_size, nhead=4, nhid=64, nlayers=2, dropout=0.1)
        self.transformer_encoder = Transformer(hidden_dim=self.hidden_size, 
                                               model_dim=256,
                                               num_heads=2, 
                                               dropout=0.1)
        
        # DiSAN
        self.conv1 = nn.Conv1d(self.hidden_size, self.hidden_size, 3, padding=1)
        self.conv2 = nn.Conv1d(self.hidden_size, self.hidden_size, 3, padding=1)
        self.fc1 = nn.Linear(2*self.hidden_size, self.rnn_hidden)

        # CHAN
        self.context_encoder = ContextAttention(self.device)

        # rnn
        self.rnn = nn.LSTM(input_size=self.hidden_size, 
                           hidden_size=self.rnn_hidden,
                           batch_first=True,
                           num_layers=1)
        
        # classifier
        self.classifier_rnn = nn.Linear(self.rnn_hidden, num_labels)
        nn.init.xavier_normal_(self.classifier_rnn.weight)
        self.classifier_bert = nn.Linear(self.hidden_size, num_labels)
        nn.init.xavier_normal_(self.classifier_bert.weight)
        self.classifier_transformer = nn.Linear(self.rnn_hidden*4, num_labels)
        nn.init.xavier_normal_(self.classifier_transformer.weight)

        # label embedding
        self.clusters = nn.Parameter(torch.randn(num_labels, config.hidden_size).float(), requires_grad=True)
        self.mapping = nn.Linear(config.hidden_size, self.rnn_hidden)

        # slot prediction
        self.slot_rnn = nn.LSTM(input_size=self.hidden_size+self.rnn_hidden,
                                hidden_size=self.rnn_hidden,
                                batch_first=True,
                                bidirectional=True,
                                num_layers=1)
        self.slot_classifier = nn.Linear(2*self.rnn_hidden, num_slot_labels)
        self.crf = CRF(self.num_slot_labels)

        # mutual iterative attention
        self.mia_encoder = MutualIterativeAttention(self.device)

        # self attentive
        self.linear1 = nn.Linear(config.hidden_size, 256)
        self.linear2 = nn.Linear(4*256, config.hidden_size)
        self.tanh = nn.Tanh()
        self.context_vector = nn.Parameter(torch.randn(256, 4), requires_grad=True)
    
    def self_attentive(self, last_hidden_states, d, b):
        # input should be (b,d,h)
        vectors = self.context_vector.unsqueeze(0).repeat(b*d, 1, 1)

        h = self.linear1(last_hidden_states) # (b*d, t, h)
        scores = torch.bmm(h, vectors) # (b*d, t, 4)
        scores = nn.Softmax(dim=1)(scores) # (b*d, t, 4)
        outputs = torch.bmm(scores.permute(0, 2, 1), h).view(b*d, -1) # (b*d, 4h)
        pooled_output = self.linear2(outputs) # (b*d, h)

        pooled_output = pooled_output.view(b,d,self.hidden_size) # (b,d,h)
        return pooled_output
    
    def mha(self, pooled_output, d, b):
        # input should be (d,b,h)
        pooled_output = pooled_output.view(d,b,self.hidden_size)
        # src_mask = self.transformer_model.generate_square_subsequent_mask(d).to(self.device)
        pooled_output = self.transformer_model(pooled_output, src_mask=None)
        pooled_output = pooled_output.view(b,d,self.hidden_size)
        return pooled_output
    
    def label_embed(self, y_caps, y_masks, rnn_out, d, b):
        last_hidden, clusters, hidden, att = self.bert(y_caps, attention_mask=y_masks)
        # clusters = self.mapping(clusters) # (n, 256)

        gram = torch.mm(clusters, clusters.permute(1,0)) # (n, n)
        rnn_out = rnn_out.reshape(b*d, self.hidden_size) # (b*d, 768)
        weights = torch.mm(rnn_out, clusters.permute(1,0)) # (b*d, n)
        logits = torch.mm(weights, torch.inverse(gram))
        logits = logits.view(b,d,self.num_labels)

        return logits
    
    def DiSAN(self, pooled_output, d, b):
        # input should be (b,h,d)
        pooled_score = pooled_output.view(b,self.hidden_size,d)
        pooled_score = torch.sigmoid(self.conv1(pooled_score))
        pooled_score = self.conv2(pooled_score)
        pooled_score = F.softmax(pooled_score, dim=-1)
        pooled_score = pooled_score.view(b,d,self.hidden_size)
        pooled_output = pooled_score * pooled_output
        return pooled_output


    def forward(self, result_ids, result_token_masks, result_masks, lengths, result_slot_labels, labels, y_caps, y_masks):
        """
        Inputs:
        result_ids:         (b, d, t)
        result_token_masks: (b, d, t)
        result_masks:       (b, d)
        lengths:            (b)
        result_slot_labels: (b, d, t)
        labels:             (b, d, l)

        BERT outputs:
        last_hidden_states: (bxd, t, h)
        pooled_output: (bxd, h), from output of a linear classifier + tanh
        hidden_states: 13 x (bxd, t, h), embed to last layer embedding
        attentions: 12 x (bxd, num_heads, t, t)
        """

        # BERT encoding
        b,d,t = result_ids.shape
        result_ids = result_ids.view(-1, t)
        result_token_masks = result_token_masks.view(-1, t)
        last_hidden_states, pooled_output, hidden_states, attentions = self.bert(result_ids, attention_mask=result_token_masks)
        pooled_output = pooled_output.view(b,d,self.hidden_size)

        ## Token: Self-attentive
        pooled_output = self.self_attentive(last_hidden_states, d, b) # (b,d,l)
        # logits = self.classifier_bert(pooled_output)
        
        ## Turn: MHA
        # pooled_output = self.mha(pooled_output, d, b) # (b,d,l)

        ## Turn: DiSAN
        # context_vector = self.DiSAN(pooled_output, d, b)
        # final_hidden = torch.cat([pooled_output, context_vector], dim=-1)
        # final_hidden = self.fc1(final_hidden)
        # logits = self.classifier_rnn(final_hidden)

        ## Turn: CHAN
        pooled_output, ffscores = self.context_encoder(pooled_output, result_masks)
        # logits = self.classifier_bert(pooled_output) # (b,d,l)

        ## Turn: transformer
        # transformer_out, attention = self.transformer_encoder(pooled_output, pooled_output, pooled_output, result_masks)
        # transformer_out = self.dropout(transformer_out)
        # logits = self.classifier_transformer(transformer_out) # (b,d,l)

        ## Prediction: RNN
        rnn_out, _ = self.rnn(pooled_output)
        rnn_out = self.dropout(rnn_out)
        logits = self.classifier_rnn(rnn_out) # (b,d,l)

        ## Prediction: Label Embedding
        # logits = self.label_embed(y_caps, y_masks, pooled_output, d, b)

        # Remove padding
        logits_no_pad = []
        labels_no_pad = []
        for i in range(b):
            logits_no_pad.append(logits[i,:lengths[i],:])
            labels_no_pad.append(labels[i,:lengths[i],:])
        logits = torch.vstack(logits_no_pad)
        labels = torch.vstack(labels_no_pad)   

        #######
        # slot prediction
        slot_vectors = last_hidden_states # (b*d,t,h)
        intent_context = rnn_out.unsqueeze(2).repeat(1,1,t,1).reshape(-1,t,self.rnn_hidden) # (b*d,t,hr)

        # comia
        # intent_context = pooled_output.unsqueeze(2)
        # slot_refined = self.mia_encoder(intent_context, slot_vectors)

        slot_inputs = torch.cat([slot_vectors, intent_context], dim=-1) # (b*d,t,h+hr)
        slot_rnn_out, _ = self.slot_rnn(slot_inputs)
        slot_rnn_out = self.dropout(slot_rnn_out)
        slot_out = self.slot_classifier(slot_rnn_out)
        slot_out = slot_out.view(-1, self.num_slot_labels) # (b*d*t, num_slots)

        # slot_loss = -self.crf(slot_out, result_slot_labels)

        return logits, labels, slot_out#, ffscores





