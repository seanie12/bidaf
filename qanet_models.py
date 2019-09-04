import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch_pretrained_bert import BertModel

dropout = 0.1


def mask_logits(inputs, mask):
    mask = mask.type(torch.float32)
    return inputs + (-1e30) * (1 - mask)


class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, relu=False,
                 stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.out = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups,
                             bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu == True:
            return F.relu(self.out(x))
        else:
            return self.out(x)


def pos_encoder(x, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1, 2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal.cuda()).transpose(1, 2)


def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                        padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


class Highway(nn.Module):
    def __init__(self, layer_num: int, size=96):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([Initialized_Conv1d(size, size, relu=False, bias=True) for _ in range(self.n)])
        self.gate = nn.ModuleList([Initialized_Conv1d(size, size, bias=True) for _ in range(self.n)])

    def forward(self, x):
        # x: shape [batch_size, hidden_size, length]
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
            # x = F.relu(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, hidden_size=96, num_head=1):
        super().__init__()
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.mem_conv = Initialized_Conv1d(hidden_size, hidden_size * 2,
                                           kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(hidden_size, hidden_size,
                                             kernel_size=1, relu=False, bias=False)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, queries, mask):
        Nh = self.num_head
        D = self.hidden_size
        memory = queries
        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        Q = self.split_last_dim(query, Nh)
        K, V = [self.split_last_dim(tensor, Nh) for tensor in torch.split(memory, D, dim=2)]

        key_depth_per_head = D // Nh
        Q *= key_depth_per_head ** -0.5
        x = self.dot_product_attention(Q, K, V, mask=mask)
        return self.combine_last_two_dim(x.permute(0, 2, 1, 3)).transpose(1, 2)

    def dot_product_attention(self, q, k, v, bias=False, mask=None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x if x != None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=dropout, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret


class Embedding(nn.Module):
    def __init__(self, embedding_size=768, hidden_size=96):
        super().__init__()
        D = hidden_size
        self.conv1d = Initialized_Conv1d(embedding_size, D, bias=False)
        self.high = Highway(2, hidden_size)

    def forward(self, wd_emb):
        wd_emb = wd_emb.transpose(1, 2)
        emb = self.conv1d(wd_emb)
        emb = self.high(emb)
        return emb


class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int, ch_num: int, k: int,
                 hidden_size: int = 96, num_head: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        D = hidden_size
        self.convs = nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = SelfAttention(hidden_size, num_head)
        self.FFN_1 = Initialized_Conv1d(ch_num, ch_num, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(ch_num, ch_num, bias=True)
        self.norm_C = nn.ModuleList([nn.LayerNorm(D) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(D)
        self.norm_2 = nn.LayerNorm(D)
        self.conv_num = conv_num

    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num + 1) * blks
        out = pos_encoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out.transpose(1, 2)).transpose(1, 2)
            if i % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, dropout * float(l) / total_layers)
            l += 1
        res = out
        out = self.norm_1(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.self_att(out, mask)
        out = self.layer_dropout(out, res, dropout * float(l) / total_layers)
        l += 1
        res = out

        out = self.norm_2(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, dropout * float(l) / total_layers)
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class CQAttention(nn.Module):
    def __init__(self, hidden_size=96):
        super().__init__()
        D = hidden_size
        w4C = torch.empty(D, 1)
        w4Q = torch.empty(D, 1)
        w4mlu = torch.empty(1, 1, D)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, C, Q, Cmask, Qmask):
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        batch_size_c = C.size()[0]
        S = self.trilinear_for_attention(C, Q)
        Lc = C.size(1)
        Lq = Q.size(1)
        Cmask = Cmask.view(batch_size_c, Lc, 1)
        Qmask = Qmask.view(batch_size_c, 1, Lq)
        S1 = F.softmax(mask_logits(S, Qmask), dim=2)
        S2 = F.softmax(mask_logits(S, Cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out.transpose(1, 2)

    def trilinear_for_attention(self, C, Q):
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        Lq = Q.size(1)
        Lc = C.size(1)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res


class Pointer(nn.Module):
    def __init__(self, hidden_size=96):
        super().__init__()
        D = hidden_size
        self.w1 = Initialized_Conv1d(D * 2, 1)
        self.w2 = Initialized_Conv1d(D * 2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        start_logits = mask_logits(self.w1(X1).squeeze(), mask)
        end_logits = mask_logits(self.w2(X2).squeeze(), mask)

        return start_logits, end_logits


class QANet(nn.Module):
    def __init__(self, ntokens, embedding="bert", embedding_size=768, hidden_size=96, num_head=1):
        super().__init__()
        if embedding == "bert":
            print("bert-embedding")
            self.word_emb = BertModel.from_pretrained("bert-base-uncased").embeddings
            for param in self.word_emb.parameters():
                param.requires_grad = False
        else:
            print("random init embedding")
            self.word_emb = nn.Embedding(ntokens, embedding_size)
        self.emb = Embedding(embedding_size, hidden_size)
        self.emb_enc = EncoderBlock(conv_num=4, ch_num=hidden_size, k=7,
                                    hidden_size=hidden_size, num_head=num_head)
        self.cq_att = CQAttention(hidden_size)
        self.cq_resizer = Initialized_Conv1d(hidden_size * 4, hidden_size)
        self.model_enc_blks = nn.ModuleList([EncoderBlock(conv_num=2, ch_num=hidden_size, k=5,
                                                          hidden_size=hidden_size) for _ in range(7)])
        self.out = Pointer(hidden_size)

    def forward(self, Cwid, Qwid, start_positions=None, end_positions=None):
        maskC = (torch.zeros_like(Cwid) != Cwid).float()
        maskQ = (torch.zeros_like(Qwid) != Qwid).float()
        Cw = self.word_emb(Cwid)
        Qw = self.word_emb(Qwid)
        C, Q = self.emb(Cw), self.emb(Qw)
        Ce = self.emb_enc(C, maskC, 1, 1)
        Qe = self.emb_enc(Q, maskQ, 1, 1)
        X = self.cq_att(Ce, Qe, maskC, maskQ)
        M0 = self.cq_resizer(X)
        M0 = F.dropout(M0, p=dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i * (2 + 2) + 1, 7)
        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i * (2 + 2) + 1, 7)
        M2 = M0
        M0 = F.dropout(M0, p=dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i * (2 + 2) + 1, 7)
        M3 = M0
        start_logits, end_logits = self.out(M1, M2, M3, maskC)

        if start_positions is not None and end_positions is not None:
            ignore_idx = start_logits.size(1)
            start_positions.clamp_(0, ignore_idx)
            end_positions.clamp_(0, ignore_idx)
            criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
            start_loss = criterion(start_logits, start_positions)
            end_loss = criterion(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

            return loss
        else:
            return start_logits, end_logits
