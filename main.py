import json
import math
import os
import re
import string
from collections import Counter

import numpy as np
import torch
import torch.cuda
import torch.nn.functional as F
import torch.optim as optim
from absl import app
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset

from config import config, device
from preproc import preproc

writer = SummaryWriter('./log1')
'''
Some functions are from the official evaluation script.
'''


class SQuADDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.context_idxs = data["context_idxs"]
        self.context_char_idxs = data["context_char_idxs"]
        self.ques_idxs = data["ques_idxs"]
        self.ques_char_idxs = data["ques_char_idxs"]
        self.y1s = data["y1s"]
        self.y2s = data["y2s"]
        self.ids = data["ids"]
        self.num = len(self.ids)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.context_idxs[idx], self.context_char_idxs[idx], self.ques_idxs[idx], self.ques_char_idxs[idx], \
               self.y1s[idx], self.y2s[idx], self.ids[idx]


class EMA(object):
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, model, num_updates):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                decay = min(self.mu, (1 + num_updates) / (10 + num_updates))
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


def collate(data):
    Cwid, Ccid, Qwid, Qcid, y1, y2, ids = zip(*data)
    Cwid = torch.tensor(Cwid).long()
    Ccid = torch.tensor(Ccid).long()
    Qwid = torch.tensor(Qwid).long()
    Qcid = torch.tensor(Qcid).long()
    y1 = torch.from_numpy(np.array(y1)).long()
    y2 = torch.from_numpy(np.array(y2)).long()
    ids = torch.from_numpy(np.array(ids)).long()
    return Cwid, Ccid, Qwid, Qcid, y1, y2, ids


def get_loader(npz_file, batch_size):
    dataset = SQuADDataset(npz_file)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=5,
                                              collate_fn=collate)
    return data_loader


def convert_tokens(eval_file, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answers"]
        prediction = value
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def train(model, optimizer, dataset, dev_dataset, dev_eval_file, start, ema):
    model.train()
    losses = []
    print(f'Training epoch {start}')
    for i, (Cwid, Ccid, Qwid, Qcid, y1, y2, ids) in enumerate(dataset):
        optimizer.zero_grad()
        # truncate un-necessary PAD token
        c_len = torch.sum(torch.sign(Cwid), 1)
        q_len = torch.sum(torch.sign(Qwid), 1)
        max_c_len = torch.max(c_len)
        max_q_len = torch.max(q_len)
        max_char_len = config.char_limit
        Cwid = Cwid[:, :max_c_len]
        Ccid = Ccid[:, :max_c_len, :max_char_len]
        Qwid = Qwid[:, :max_q_len]
        Qcid = Qcid[:, :max_q_len, :max_char_len]

        Cwid, Ccid, Qwid, Qcid = Cwid.to(device), Ccid.to(device), Qwid.to(device), Qcid.to(device)
        p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
        y1, y2 = y1.to(device), y2.to(device)
        p1 = F.log_softmax(p1, dim=1)
        p2 = F.log_softmax(p2, dim=1)
        loss1 = F.nll_loss(p1, y1)
        loss2 = F.nll_loss(p2, y2)
        loss = (loss1 + loss2) / 2
        writer.add_scalar('data/loss', loss.item(), i + start * len(dataset))
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), config.grad_clip)
        optimizer.step()

        ema(model, i + start * len(dataset))

        if (i + 1) % config.checkpoint == 0 and (i + 1) < config.checkpoint * (len(dataset) // config.checkpoint):
            ema.assign(model)
            test(model, dev_dataset, dev_eval_file, i + start * len(dataset))
            ema.resume(model)
            model.train()
        for param_group in optimizer.param_groups:
            writer.add_scalar('data/lr', param_group['lr'], i + start * len(dataset))
        print("\rSTEP {:8d}/{} loss {:8f}".format(i + 1, len(dataset), loss.item()), end='')
    loss_avg = np.mean(losses)
    print("STEP {:8d} Avg_loss {:8f}\n".format(start, loss_avg))


def test(model, dataset, eval_file, test_i):
    print("\nTest")
    model.eval()
    answer_dict = {}
    losses = []
    with torch.no_grad():
        for i, (Cwid, Ccid, Qwid, Qcid, y1, y2, ids) in enumerate(dataset):
            Cwid, Ccid, Qwid, Qcid = Cwid.to(device), Ccid.to(device), Qwid.to(device), Qcid.to(device)

            start_logits, end_logits = model(Cwid, Ccid, Qwid, Qcid)
            y1, y2 = y1.to(device), y2.to(device)
            p1 = F.log_softmax(start_logits, dim=1)
            p2 = F.log_softmax(end_logits, dim=1)
            loss1 = F.nll_loss(p1, y1)
            loss2 = F.nll_loss(p2, y2)
            loss = torch.mean(loss1 + loss2)
            losses.append(loss.item())

            p1 = F.softmax(start_logits, dim=1)
            p2 = F.softmax(end_logits, dim=1)

            outer = torch.matmul(p1.unsqueeze(2), p2.unsqueeze(1))
            for j in range(outer.size()[0]):
                outer[j] = torch.triu(outer[j])
            a1, _ = torch.max(outer, dim=2)
            a2, _ = torch.max(outer, dim=1)
            ymin = torch.argmax(a1, dim=1)
            ymax = torch.argmax(a2, dim=1)

            answer_dict_, _ = convert_tokens(eval_file, ids.tolist(), ymin.tolist(), ymax.tolist())
            answer_dict.update(answer_dict_)
            print("\rSTEP {:8d}/{} loss {:8f}".format(i + 1, len(dataset), loss.item()), end='')

    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    f = open("log/answers.json", "w")
    json.dump(answer_dict, f)
    f.close()
    metrics["loss"] = loss
    print("EVAL loss {:8f} F1 {:8f} EM {:8f}\n".format(loss, metrics["f1"], metrics["exact_match"]))
    if config.mode == "train":
        writer.add_scalar('data/test_loss', loss, test_i)
        writer.add_scalar('data/F1', metrics["f1"], test_i)
        writer.add_scalar('data/EM', metrics["exact_match"], test_i)
    return metrics


def train_entry(config):
    from models import BiDAF

    with open(config.word_emb_file, "rb") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "rb") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)

    print("Building model...")

    train_dataset = get_loader(config.train_record_file, config.batch_size)
    dev_dataset = get_loader(config.dev_record_file, config.batch_size)

    c_vocab_size, c_emb_size = char_mat.shape
    model = BiDAF(word_mat, w_embedding_size=300, c_embeding_size=c_emb_size,
                  c_vocab_size=c_vocab_size, hidden_size=100, drop_prob=0.2).to(device)
    if config.pretrained:
        print("load pre-trained model")
        state_dict = torch.load(config.save_path, map_location="cpu")
        model.load_state_dict(state_dict)

    ema = EMA(config.decay)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = optim.Adadelta(lr=0.5, params=parameters)
    best_f1 = 0
    best_em = 0
    patience = 0
    for iter in range(config.num_epoch):
        train(model, optimizer, train_dataset, dev_dataset, dev_eval_file, iter, ema)
        ema.assign(model)
        metrics = test(model, dev_dataset, dev_eval_file, (iter + 1) * len(train_dataset))
        dev_f1 = metrics["f1"]
        dev_em = metrics["exact_match"]
        if dev_f1 < best_f1 and dev_em < best_em:
            patience += 1
            if patience > config.early_stop:
                break
        else:
            patience = 0
            best_f1 = max(best_f1, dev_f1)
            best_em = max(best_em, dev_em)

        fn = os.path.join(config.save_dir, "model_{}_{:.2f}_{:.2f}.pt".format(iter, best_f1, best_em))
        torch.save(model.state_dict(), fn)
        ema.resume(model)


def test_entry(config):
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    dev_dataset = get_loader(config.dev_record_file, config.batch_size)
    fn = os.path.join(config.save_dir, "model.pt")
    model = torch.load(fn)
    test(model, dev_dataset, dev_eval_file, 0)


def main(_):
    if config.mode == "train":
        train_entry(config)
    elif config.mode == "data":
        preproc(config)
    else:
        exit()


if __name__ == '__main__':
    app.run(main)
