from pytorch_pretrained_bert import BertTokenizer
from models import BiDAF
import argparse
import torch
from utils import EMA, get_data_loader, cal_running_avg_loss, progress_bar, eta, get_ext_data_loader
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.nn as nn
import time
from eval import eval_qa
import os


def main(args):

    save_dir = os.path.join("./save", time.strftime("%m%d%H%M%S"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if args.all_data:
        data_loader = get_ext_data_loader(tokenizer, "./data/train/", shuffle=True, args=args)
    else:
        data_loader, _, _ = get_data_loader(tokenizer, "./data/train-v1.1.json",
                                            shuffle=True, args=args)
    vocab_size = len(tokenizer.vocab)
    model = BiDAF(embedding_size=args.embedding_size,
                  vocab_size=vocab_size,
                  hidden_size=args.hidden_size,
                  drop_prob=args.dropout)
    if args.load_model:
        state_dict = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("load pre-trained model")
    device = torch.device("cuda")
    model = model.to(device)
    model.train()
    ema = EMA(model, args.decay)

    optimizer = optim.Adam(model.parameters(), args.lr)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)
    step = 0
    num_batches = len(data_loader)
    avg_loss = 0
    best_f1 = 0
    for epoch in range(1, args.num_epochs + 1):
        step += 1
        start = time.time()
        model.train()
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
            loss = model(c_ids, q_ids,
                         start_positions=start_positions,
                         end_positions=end_positions)
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
        metric_dict = eval_qa(args, model)
        f1 = metric_dict["f1"]
        em = metric_dict["exact_match"]
        print("epoch: {}, final loss: {:.4f}, F1:{:.2f}, EM:{:.2f}"
              .format(epoch, avg_loss, f1, em))

        if f1 > best_f1:
            best_f1 = f1
            state_dict = model.state_dict()
            save_file = "bidaf_{:.2f}_{:.2f}".format(f1, em)
            path = os.path.join(save_dir, save_file)
            torch.save(state_dict, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_size", default=100, type=int, help="embedding_size")
    parser.add_argument("--hidden_size", default=100, type=int, help="hidden size")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--dropout", default=0.2, type=float, help="dropout")
    parser.add_argument("--decay", default=0.999, type=float, help="exp moving average decay")
    parser.add_argument("--num_epochs", default=20, type=int, help="num epochs")
    parser.add_argument("--batch_size", default=60, type=int, help="batch_size")
    parser.add_argument("--max_seq_len", default=384, type=int, help="max context length")
    parser.add_argument("--max_query_len", default=64, type=int, help="max query length")
    parser.add_argument("--doc_stride", default=128, type=int)
    parser.add_argument("--weight_decay", default=0, type=float, help="weight decay")
    parser.add_argument("--max_grad_norm", default=5.0, type=float, help="max grad norm")
    parser.add_argument("--debug", action="store_true", help="whether to debug")
    parser.add_argument("--all_data", action="store_true")
    parser.add_argument("--files", type=str, default="squad_naturalquestions")
    parser.add_argument("--model_path", type=str, default="./save/bidaf_71", help="model path")
    parser.add_argument("--load_model", action="store_true")
    args = parser.parse_args()
    main(args)
