from pytorch_pretrained_bert import BertTokenizer
from models import BiDAF
import argparse
import torch
from utils import EMA, get_data_loader, cal_running_avg_loss, progress_bar, eta
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.nn as nn
import time


def main(args):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    data_loader, _, _ = get_data_loader(tokenizer, "./data/train-v1.1.json",
                                        shuffle=True, args=args)

    model = BiDAF(768, args.hidden_size, args.dropout)
    device = torch.device("cuda")
    model = model.to(device)
    model.train()
    ema = EMA(model, args.decay)

    optimizer = optim.Adadelta(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)
    step = 0
    num_batches = len(data_loader)
    avg_loss = 0
    for epoch in range(1, args.num_epochs + 1):
        step += 1
        start = time.time()
        for i, batch in enumerate(data_loader, start=1):
            c_ids, q_ids, start_positions, end_positions = batch
            c_len = torch.sum(torch.sign(c_ids), 1)
            max_c_len = torch.max(c_len)
            c_ids = c_ids[:, :max_c_len].to(device)
            q_len = torch.sum(torch.sign(q_ids), 1)
            max_q_len = torch.max(q_len)
            q_ids = q_ids[:, :max_q_len].to(device)

            start_positions = start_positions.to(device)
            end_positions = end_positions.to(device)

            optimizer.zero_grad()
            loss = model(c_ids, q_ids, start_positions, end_positions)
            loss.backward()
            avg_loss = cal_running_avg_loss(loss.item(), avg_loss)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step(step // args.batch_size)
            ema(model, step // args.batch_size)

            step += args.batch_size

            msg = "{}/{} {} - ETA : {} - qa_loss: {:.4f}" \
                .format(i, num_batches, progress_bar(i, num_batches),
                        eta(start, i, num_batches),
                        avg_loss)
            print(msg, end="\r")
        print("epoch: {}, final loss: {:.4f}".format(epoch, avg_loss))
    state_dict = model.state_dict()
    save_file = "./save/bidaf_{}".format(avg_loss)
    torch.save(state_dict, save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", default=100, type=int, help="hidden size")
    parser.add_argument("--lr", default=0.5, type=float, help="learning rate")
    parser.add_argument("--dropout", default=0.2, type=float, help="dropout")
    parser.add_argument("--decay", default=0.999, type=float, help="exp moving average decay")
    parser.add_argument("--num_epochs", default=12, type=int, help="num epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="batch_size")
    parser.add_argument("--max_seq_len", default=400, type=int, help="max context length")
    parser.add_argument("--max_query_len", default=64, type=int, help="max query length")
    parser.add_argument("--weight_decay", default=0, type=float, help="weight decay")
    parser.add_argument("--max_grad_norm", default=5.0, type=float, help="max grad norm")
    parser.add_argument("--debug", action="store_true", help="whether to debug")
    args = parser.parse_args()
    main(args)
