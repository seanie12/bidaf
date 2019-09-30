import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from typing import Optional


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.
    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).
    Args:
        word-mat (numpy array): Pre-trained word vectors.
        w_embedding_size: word embedding_size
        c_embedding_size: char_embedding_size
        c_vocab_size: the number of characters
        char_dim (int): Size of hidden activations.
        dropout_p (float): Probability of zero-ing out activations
    """

    def __init__(self, word_mat, w_embedding_size,
                 c_embedding_size, c_vocab_size, char_dim, dropout_p=0.2):
        super(Embedding, self).__init__()

        self.w_embed = nn.Embedding.from_pretrained(torch.Tensor(word_mat), freeze=True)
        self.c_embed = nn.Embedding(c_vocab_size, c_embedding_size, padding_idx=0)

        self.conv2d = nn.Conv2d(c_embedding_size, char_dim, kernel_size=(1, 5))
        nn.init.kaiming_uniform_(self.conv2d.weight, nonlinearity="relu")

        self.dropout = nn.Dropout(dropout_p)
        # self.hwy = HighwayEncoder(2, 2 * w_embedding_size)

    def forward(self, word_id, char_id):
        w_emb = self.w_embed(word_id)  # (batch_size, seq_len, embed_size)
        # w_emb = self.dropout(w_emb)

        char_emb = self.c_embed(char_id)
        # char_emb = self.dropout(char_emb)
        char_emb = char_emb.permute(0, 3, 1, 2)
        char_emb = F.relu(self.conv2d(char_emb))
        char_emb, _ = torch.max(char_emb, dim=3)  # max-pooling
        char_emb = char_emb.squeeze()
        char_emb = char_emb.transpose(1, 2)

        emb = torch.cat([w_emb, char_emb], dim=-1)
        # emb = self.hwy(emb)  # (batch_size, seq_len, 2 *hidden_size)

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.
    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).
    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """

    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.
    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.
    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True,
                          bidirectional=True,
                          dropout=drop_prob if num_layers > 1 else 0.)
        self.var_dropout = VariationalDropout(drop_prob, batch_first=True)
        # self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)
        x = self.var_dropout(x)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]  # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]  # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        # x = self.dropout(x)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.
    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).
    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)  # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        #
        _s1 = s.masked_fill(q_mask, value=-1e30)
        _s2 = s.masked_fill(c_mask, value=-1e30)
        s1 = F.softmax(_s1, dim=2)  # (batch_size, c_len, q_len)
        s2 = F.softmax(_s2, dim=1)  # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.matmul(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.matmul(torch.matmul(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).
        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.
        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2) \
            .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFSelfAttention(nn.Module):
    def __init__(self, num_head, hidden_size, dropout=0.1, var_dropout=0.2):
        super(BiDAFSelfAttention, self).__init__()
        assert hidden_size % num_head == 0
        self.d_k = hidden_size // num_head
        self.linear_lst = self.clones(nn.Linear(hidden_size, hidden_size), 4)
        self.var_dropout = VariationalDropout(var_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.h = num_head

    @staticmethod
    def clones(module, num_layers):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        """

        :param query: [b,num_head, t, d_k]
        :param key: [b,num_head, t, d_k]
        :param value: [b,num_head, t, d_k]
        :param mask: [b,1, t] 1 for PAD 0 for the others
        :param dropout: nn.Dropout()
        :return: QK^TV / d_k^0.5
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # a_{ij} = -inf if i = j as https://www.aclweb.org/anthology/P18-1078
        q_t, k_t = query.size(2), key.size(2)
        diag = torch.eye(q_t, k_t, device=query.device).byte()
        expanded_diag = diag.expand_as(scores)
        scores.masked_fill(expanded_diag, -1e9)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_lst, (query, key, value))]
        x = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linear_lst[-1](x)


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.
    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.
    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask, start_positions=None, end_positions=None):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        length = torch.sum((1 - mask.float()), 1)
        mod_2 = self.rnn(mod, length)
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        masked_start_logits = logits_1.squeeze().masked_fill(mask, -1e30)
        masked_end_logits = logits_2.squeeze().masked_fill(mask, -1e30)

        if start_positions is not None and end_positions is not None:
            ignore_idx = masked_start_logits.size(1)
            start_positions.clamp_(0, ignore_idx)
            end_positions.clamp_(0, ignore_idx)
            criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
            start_loss = criterion(masked_start_logits, start_positions)
            end_loss = criterion(masked_end_logits, end_positions)
            return (start_loss + end_loss) / 2

        else:
            return masked_start_logits, masked_end_logits


class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """

    def __init__(self, dropout: float, batch_first: Optional[bool] = False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty((max_batch_size, 1, x.size(2)), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty((1, max_batch_size, x.size(2)), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        return x
