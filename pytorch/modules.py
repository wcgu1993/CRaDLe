import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import optim
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)

class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)  # (batch_size, hidden_size, 1)
                            )

        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=True).cuda()
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row

        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions


class TokenEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, blo_len, n_layers=1):
        super(TokenEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.blo_len = blo_len
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.test = nn.Linear(emb_size, hidden_size * 2)
        self.pos_mapping = nn.Linear(blo_len, emb_size)
        self.lstm = nn.LSTM(emb_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.attn_layer = nn.Linear(emb_size, 1)
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[0], 0)
        nn.init.uniform_(self.attn_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.attn.weight, -0.1, 0.1)
        nn.init.uniform_(self.pos_mapping.weight, -0.1, 0.1)
        nn.init.uniform_(self.test.weight, -0.1, 0.1)
        for name, param in self.lstm.named_parameters(): # initialize the gate weights
            if 'weight' in name or 'bias' in name:
                param.data.uniform_(-0.1, 0.1)


    def forward(self, tokens, index):

        batch_size, blo_len, seq_len = tokens.size()
        tokens = tokens.long()
        index = index.float()
        embedded = self.embedding(tokens)  # tokens: [batch_sz x bol_len x seq_len x 1]  embedded: [batch_sz x bol_len x seq_len x emb_sz]
        embedded_inter = self.attn_layer(embedded)

        token_score = torch.exp(embedded_inter.squeeze(3))
        token_weight = token_score / torch.sum(token_score, 2).unsqueeze(2)
        output_pool = torch.einsum('ijk,ijkl->ijl', token_weight, embedded)

        pos_embedded = torch.bmm(index, output_pool)
        blo_sum = torch.sum(index, 2)
        blo_sum = blo_sum.masked_fill(blo_sum == 0.0, 1.0)
        blo_sum = blo_sum.unsqueeze(2)
        pos_embedded = pos_embedded / blo_sum
        combined_embedded = torch.cat((output_pool, pos_embedded), 2) # [batch_size x  bol_len x (emb_size + bol_len)]
        hids, (h_n, c_n) = self.lstm(combined_embedded)
        
        h_n = h_n.view(self.n_layers, 2, batch_size, self.hidden_size)  # [n_layers x n_dirs x batch_sz x hid_sz]
        h_n = h_n[-1]  # get the last layer [n_dirs x batch_sz x hid_sz]
        encoding = h_n.view(batch_size, -1)  # [batch_sz x (n_dirs*hid_sz)]
        return encoding  # pooled_encoding
        
class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, n_layers=1):
        super(SeqEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.init_weights()
        
    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[0], 0)
        for name, param in self.lstm.named_parameters(): # initialize the gate weights
            if 'weight' in name or 'bias' in name: 
                param.data.uniform_(-0.1, 0.1)

    def hids_masked(self, inputs, mask=None, dim=1):
        if mask is None:
            return input
        else:
            mask = mask.unsqueeze(-1)
            mask = mask.repeat(1, 1, input.size(-1))
            inputs = inputs.masked_fill(mask == 0.0, 0.0)
            return inputs

    def forward(self, inputs, input_lens=None): 
        batch_size, seq_len=inputs.size()
        inputs = self.embedding(inputs)  # input: [batch_sz x seq_len]  embedded: [batch_sz x seq_len x emb_sz]
        inputs = F.dropout(inputs, 0.25, self.training)
        
        if input_lens is not None:# sort and pack sequence 
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)
            
        hids, (h_n, c_n) = self.lstm(inputs) # hids:[b x seq x hid_sz*2](biRNN) 
        
        if input_lens is not None: # reorder and pad
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)   
            hids = F.dropout(hids, p=0.25, training=self.training)
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
        h_n = h_n.view(self.n_layers, 2, batch_size, self.hidden_size) #[n_layers x n_dirs x batch_sz x hid_sz]
        h_n = h_n[-1] # get the last layer [n_dirs x batch_sz x hid_sz]
        encoding = h_n.view(batch_size,-1) #[batch_sz x (n_dirs*hid_sz)]

        return encoding #pooled_encoding

    
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)    
    

def get_word_weights(vocab_size, padding_idx=0):
    '''contruct a word weighting table '''
    def cal_weight(word_idx):
        return 1-math.exp(-word_idx)
    weight_table = np.array([cal_weight(w) for w in range(vocab_size)])
    if padding_idx is not None:        
        weight_table[padding_idx] = 0. # zero vector for padding dimension
    return torch.FloatTensor(weight_table)

 
 
 
 
