"""Top-level model classes.
Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn

import layers


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.
    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).
    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.
    Args:
        embedding_size (int): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, word_mat, w_embedding_size, c_embeding_size, c_vocab_size,
                 hidden_size, num_head=1, drop_prob=0.2):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_mat, w_embedding_size,
                                    c_embeding_size,
                                    c_vocab_size,
                                    hidden_size, drop_prob)
        self.enc = layers.RNNEncoder(input_size=w_embedding_size + hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)
        self.var_dropout = layers.VariationalDropout(drop_prob, batch_first=True)
        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)
        self.linear_trans = nn.Sequential(nn.Linear(8 * hidden_size, 2 * hidden_size),
                                          nn.ReLU())
        self.attn_mod = layers.RNNEncoder(hidden_size * 2, hidden_size,
                                          num_layers=1,
                                          drop_prob=drop_prob)

        self.self_attn = layers.BiDAFSelfAttention(num_head, 2 * hidden_size)
        self.linear_attn = nn.Sequential(nn.Linear(2 * hidden_size, 2 * hidden_size), nn.ReLU())

        self.mod = layers.RNNEncoder(input_size=2 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, cc_idx, qw_idxs, qc_idx, start_positions=None, end_positions=None):
        c_mask = (cw_idxs == 0).byte()  # 1 for PAD 0 for the others
        c_len = torch.sum((1 - c_mask.long()), 1)

        q_mask = (qw_idxs == 0).byte()  # 1 for PAD 0 for the others
        q_len = torch.sum((1 - q_mask.long()), 1)

        c_emb = self.emb(cw_idxs, cc_idx)  # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idx)  # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)  # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)  # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)  # (batch_size, c_len, 8 * hidden_size)

        att_trans = self.linear_trans(att)
        att_mod = self.attn_mod(att_trans, c_len)

        # self-attention layer
        self_attn_mask = c_mask.unsqueeze(2)
        att_mod = self.var_dropout(att_mod)
        self_attn = self.self_attn(att_mod, att_mod, att_mod, self_attn_mask)
        self_attn = self.linear_attn(self_attn)
        res_sum = att_trans + self_attn

        mod = self.mod(res_sum, c_len)  # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask, start_positions, end_positions)  # 2 tensors, each (batch_size, c_len)

        return out

    def kl(self):
        kl = 0
        for name, module in self.named_modules():
            if isinstance(module, layers.VariationalDropout):
                kl += module.kl().sum()
        return kl
