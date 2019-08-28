from squad_utils import read_squad_examples, convert_examples_to_features_answer_id, read_examples
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import time


class EMA(object):
    """Exponential moving average of model parameters.
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


def get_data_loader(tokenizer, file, shuffle, args):
    examples = read_squad_examples(file, is_training=True, debug=args.debug)
    features = convert_examples_to_features_answer_id(examples,
                                                      tokenizer=tokenizer,
                                                      max_seq_length=args.max_seq_len,
                                                      max_query_length=args.max_query_len,
                                                      doc_stride=128,
                                                      is_training=True)

    all_c_ids = torch.tensor([f.c_ids for f in features], dtype=torch.long)
    all_q_ids = torch.tensor([f.q_ids for f in features], dtype=torch.long)
    all_start_positions = torch.tensor([f.noq_start_position for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f.noq_end_position for f in features], dtype=torch.long)

    all_data = TensorDataset(all_c_ids, all_q_ids, all_start_positions, all_end_positions)
    data_loader = DataLoader(all_data, args.batch_size, shuffle=shuffle)

    return data_loader, examples, features


def get_ext_data_loader(tokenizer, train_folder, shuffle, args):
    train_files = [f for f in os.listdir(train_folder) if f.endswith(".gz")]
    examples_lst = []
    features_lst = []
    file_list = args.files.split("_")
    for file in train_files:
        file_name = file.split(".")[0].lower()
        if file_name not in file_list:
            print("skip {}".format(file_name))
            continue
        print("preprocessing {}".format(file_name))
        file_path = os.path.join(train_folder, file)
        examples = read_examples(file_path, debug=args.debug)
        examples_lst.extend(examples)

        features = convert_examples_to_features_answer_id(examples,
                                                          tokenizer=tokenizer,
                                                          max_seq_length=args.max_seq_len,
                                                          max_query_length=args.max_query_len,
                                                          doc_stride=args.doc_stride,
                                                          is_training=True)
        features_lst.extend(features)
    all_c_ids = torch.tensor([f.c_ids for f in features_lst], dtype=torch.long)
    all_q_ids = torch.tensor([f.q_ids for f in features_lst], dtype=torch.long)

    all_noq_start_positions = torch.tensor([f.noq_start_position for f in features_lst], dtype=torch.long)
    all_noq_end_positions = torch.tensor([f.noq_end_position for f in features_lst], dtype=torch.long)

    train_data = TensorDataset(all_c_ids, all_q_ids, all_noq_start_positions, all_noq_end_positions)

    data_loader = DataLoader(train_data, shuffle=shuffle, batch_size=args.batch_size)
    return data_loader


def time_since(t):
    """ Function for time. """
    return time.time() - t


def progress_bar(completed, total, step=5):
    """ Function returning a string progress bar. """
    percent = int((completed / total) * 100)
    bar = '[='
    arrow_reached = False
    for t in range(step, 101, step):
        if arrow_reached:
            bar += ' '
        else:
            if percent // t != 0:
                bar += '='
            else:
                bar = bar[:-1]
                bar += '>'
                arrow_reached = True
    if percent == 100:
        bar = bar[:-1]
        bar += '='
    bar += ']'
    return bar


def user_friendly_time(s):
    """ Display a user friendly time from number of second. """
    s = int(s)
    if s < 60:
        return "{}s".format(s)

    m = s // 60
    s = s % 60
    if m < 60:
        return "{}m {}s".format(m, s)

    h = m // 60
    m = m % 60
    if h < 24:
        return "{}h {}m {}s".format(h, m, s)

    d = h // 24
    h = h % 24
    return "{}d {}h {}m {}s".format(d, h, m, s)


def eta(start, completed, total):
    """ Function returning an ETA. """
    # Computation
    took = time_since(start)
    time_per_step = took / completed
    remaining_steps = total - completed
    remaining_time = time_per_step * remaining_steps

    return user_friendly_time(remaining_time)


def cal_running_avg_loss(loss, running_avg_loss, decay=0.99):
    if running_avg_loss == 0:
        return loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    return running_avg_loss
