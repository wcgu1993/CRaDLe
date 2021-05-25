import sys
import torch 
import torch.utils.data as data
import torch.nn as nn
import tables
import json
import random
import numpy as np
import pickle
from utils import PAD_ID, SOS_ID, EOS_ID, UNK_ID, indexes2sent

    
class CodeSearchDataset(data.Dataset):
    """
    Dataset that has only positive samples.
    """
    def __init__(self, data_dir,
                 f_tokens, max_tok_len, f_index, max_index_len, f_block, max_block_len, f_descs = None, max_desc_len = None):
        self.max_tok_len = max_tok_len
        self.max_desc_len = max_desc_len
        self.max_block_len = max_block_len
        # 1. Initialize file path or list of file names.
        """read training data(list of int arrays) from a hdf5 file"""
        self.training=False
        print("loading data...")
        table_tokens = tables.open_file(data_dir+f_tokens)
        self.tokens = table_tokens.get_node('/phrases')[:].astype(np.long)
        self.idx_tokens = table_tokens.get_node('/indices')[:]
        table_index = tables.open_file(data_dir+f_index)
        self.index = table_index.get_node('/phrases')[:].astype(np.long)
        self.idx_index = table_index.get_node('/indices')[:]
        table_tok_blocks = tables.open_file(data_dir+f_block)
        self.tok_blocks = table_tok_blocks.get_node('/indices')[:]
        if f_descs is not None:
            self.training = True
            table_desc = tables.open_file(data_dir+f_descs)
            self.descs = table_desc.get_node('/phrases')[:].astype(np.long)
            self.idx_descs = table_desc.get_node('/indices')[:]

        if f_descs is not None:
            assert self.tok_blocks.shape[0]==self.idx_descs.shape[0]
        self.data_len = self.tok_blocks.shape[0]
        print("{} entries".format(self.data_len))
        
    def pad_seq(self, seq, maxlen):
        if len(seq)<maxlen:
            # !!!!! numpy appending is slow. Try to optimize the padding
            seq=np.append(seq, [PAD_ID]*(maxlen-len(seq)))
        seq=seq[:maxlen]
        return seq
    
    def __getitem__(self, offset):
        block_len, block_pos = self.tok_blocks[offset]['length'], self.tok_blocks[offset]['pos']
        block_len = min(int(block_len), self.max_block_len)
        start_pos = block_pos
        tokens = np.zeros((self.max_block_len, self.max_tok_len))
        index = np.zeros((self.max_block_len, self.max_block_len))

        for num in range(block_len):
            index_len, index_pos = self.idx_index[start_pos]['length'], self.idx_index[start_pos]['pos']
            index_len = min(int(index_len), self.max_block_len)
            single_index = self.index[index_pos:index_pos+index_len]
            for i in single_index:
                if i > 0:
                    index[num][i] = 1
            token_len, token_pos = self.idx_tokens[start_pos]['length'], self.idx_tokens[start_pos]['pos']
            token_len = min(int(token_len), self.max_tok_len)
            single_token = self.tokens[token_pos:token_pos+token_len]
            single_token = self.pad_seq(single_token, self.max_tok_len)
            tokens[num] = single_token
            start_pos += 1

        if self.training:
            len1, pos = self.idx_descs[offset]['length'], self.idx_descs[offset]['pos']
            good_desc_len = min(int(len1), self.max_desc_len)
            good_desc = self.descs[pos:pos+good_desc_len]
            good_desc = self.pad_seq(good_desc, self.max_desc_len)
            
            rand_offset = random.randint(0, self.data_len-1)
            len1, pos = self.idx_descs[rand_offset]['length'], self.idx_descs[rand_offset]['pos']
            bad_desc_len = min(int(len1), self.max_desc_len)
            bad_desc = self.descs[pos:pos+bad_desc_len]
            bad_desc = self.pad_seq(bad_desc, self.max_desc_len)

            return tokens, index, good_desc, good_desc_len, bad_desc, bad_desc_len
        return tokens, index

        
    def __len__(self):
        return self.data_len
    

def load_dict(filename):
    return json.loads(open(filename, "r").readline())
    #return pickle.load(open(filename, 'rb')) 

def load_vecs(fin):         
    """read vectors (2D numpy array) from a hdf5 file"""
    h5f = tables.open_file(fin)
    h5vecs = h5f.root.vecs
    
    vecs = np.zeros(shape=h5vecs.shape,dtype=h5vecs.dtype)
    vecs[:] = h5vecs[:]
    h5f.close()
    return vecs
        
def save_vecs(vecs, fout):
    fvec = tables.open_file(fout, 'w')
    atom = tables.Atom.from_dtype(vecs.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = fvec.create_carray(fvec.root,'vecs', atom, vecs.shape,filters=filters)
    ds[:] = vecs
    print('done')
    fvec.close()

if __name__ == '__main__':
    input_dir='./data/github/'
    train_set = CodeSearchDataset(input_dir, 'train.ordered.tokens.h5', 30, 'train.desc.h5', 30)
    train_data_loader=torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=False, num_workers=1)
    use_set = CodeSearchDataset(input_dir, 'use.ordered.tokens.h5', 30, 'use.desc.h5', 30)
    use_data_loader=torch.utils.data.DataLoader(dataset=use_set, batch_size=1, shuffle=False, num_workers=1)
    vocab_tokens = load_dict(input_dir+'vocab.tokens.json')
    vocab_desc = load_dict(input_dir+'vocab.desc.json')
    
    print('============ Train Data ================')
    k=0
    for batch in train_data_loader:
        batch = tuple([t.numpy() for t in batch])
        tokens, tok_len, good_desc, good_desc_len, bad_desc, bad_desc_len = batch
        k+=1
        if k>20: break
        print('-------------------------------')
        print(indexes2sent(tokens, vocab_tokens))
        print(indexes2sent(good_desc, vocab_desc))
        
    print('\n\n============ Use Data ================')
    k=0
    for batch in use_data_loader:
        batch = tuple([t.numpy() for t in batch])
        tokens, tok_len = batch
        k+=1
        if k>20: break
        print('-------------------------------')
        print(indexes2sent(tokens, vocab_tokens))
