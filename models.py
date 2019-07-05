"""Top-level model classes.
Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, embedding_size, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(embedding_size,
                                    hidden_size=hidden_size)

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

    def forward(self, cw_idxs, qw_idxs, start_positions=None, end_positions=None):
        c_mask = (cw_idxs == 0).byte()  # 1 for PAD 0 for the others
        q_mask = (qw_idxs == 0).byte()  # 1 for PAD 0 for the others
        c_len = torch.sum((1 - c_mask.long()), 1)
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


class LM(nn.Module):
    def __init__(self, embed_dim, max_seq_len, num_rep, vocab_size,
                 padding_idx, dis_filter_sizes, dis_num_filters, dropout=0.25):
        super(LM, self).__init__()

        self.embed_dim = embed_dim  # 1
        self.max_seq_len = max_seq_len
        self.feature_dim = sum(dis_num_filters)
        self.emb_dim_single = int(embed_dim / num_rep)
        self.vocab_size = vocab_size
        self.embeddings = nn.Linear(vocab_size, embed_dim, bias=False)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, self.emb_dim_single), stride=(1, self.emb_dim_single)) for (n, f) in
            zip(dis_num_filters, dis_filter_sizes)
        ])

        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 100)
        self.out2logits = nn.Linear(100, 1)
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    # inp = F.one_hot(real_samples, cfg.vocab_size).float()
    def forward(self, inp):
        """
        Get logits of discriminator
        :param inp: [batch_size, seq_len, vocab_size]
        :return logits: [batch_size, num_rep] (1-D tensor)
        """
        if len(inp.size()) < 3:
            inp = F.one_hot(inp, self.vocab_size)
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim

        cons = [F.relu(conv(emb)) for conv in self.convs]  # [batch_size * num_filter * (seq_len-k_h+1) * num_rep]
        pools = [F.max_pool2d(con, (con.size(2), 1)).squeeze(2) for con in cons]  # [batch_size * num_filter * num_rep]
        pred = torch.cat(pools, 1)
        pred = pred.permute(0, 2, 1).contiguous().view(-1, self.feature_dim)  # (batch_size * num_rep) * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        pred = self.feature2out(self.dropout(pred))
        logits = self.out2logits(pred).squeeze(1)  # [batch_size * num_rep]

        return logits
