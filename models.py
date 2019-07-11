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

    def __init__(self, embedding_size, vocab_size, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(embedding_size, vocab_size,
                                    hidden_size, drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, q_len=None, start_positions=None, end_positions=None):
        c_mask = (cw_idxs == 0).byte()  # 1 for PAD 0 for the others
        c_len = torch.sum((1 - c_mask.long()), 1)

        if q_len is not None:
            # in case qw_idxs are vocab_dist [b,t,|V|]
            batch_size, nsteps, _ = qw_idxs.size()
            indices = torch.arange(nsteps, device=qw_idxs.device).repeat([batch_size, 1])
            reverse_mask = indices < q_len.unsqueeze(1)  # 0 for PAD
            q_mask = (1 - reverse_mask).byte()  # 1 for PAD
        else:
            q_mask = (qw_idxs == 0).byte()  # 1 for PAD 0 for the others
            q_len = torch.sum((1 - q_mask.long()), 1)

        c_emb = self.emb(cw_idxs)  # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)  # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)  # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)  # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)  # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)  # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask, start_positions, end_positions)  # 2 tensors, each (batch_size, c_len)

        return out
