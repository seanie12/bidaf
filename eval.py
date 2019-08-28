import argparse
import collections
import json

import torch
from pytorch_pretrained_bert import BertTokenizer

from models import BiDAF
from squad_utils import write_predictions, evaluate
from utils import get_data_loader


def eval_qa(args, model=None):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    eval_loader, eval_examples, eval_features = get_data_loader(tokenizer, "./data/dev-v1.1.json",
                                                                shuffle=False, args=args)
    if model is None:
        vocab_size = len(tokenizer.vocab)
        model = BiDAF(embedding_size=100,
                      vocab_size=vocab_size,
                      hidden_size=args.hidden_size,
                      drop_prob=0.0)
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict)
    model.eval()
    device = torch.device("cuda")
    model.to(device)

    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])
    model.eval()
    all_results = []
    example_index = -1
    for batch in eval_loader:
        c_ids, q_ids, _, _ = batch
        c_len = torch.sum(torch.sign(c_ids), 1)
        q_len = torch.sum(torch.sign(q_ids), 1)
        max_c_len = torch.max(c_len)
        max_q_len = torch.max(q_len)
        c_ids = c_ids[:, :max_c_len].to(device)
        q_ids = q_ids[:, :max_q_len].to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(c_ids, q_ids)
            batch_size = batch_start_logits.size(0)
        for i in range(batch_size):
            example_index += 1
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))

    output_prediction_file = "./result/bidaf_pred.json"
    write_predictions(eval_examples, eval_features, all_results, n_best_size=20,
                      max_answer_length=30, do_lower_case=True,
                      output_prediction_file=output_prediction_file,
                      verbose_logging=False,
                      version_2_with_negative=False,
                      null_score_diff_threshold=0,
                      noq_position=True)

    with open("data/dev-v1.1.json") as f:
        dataset_json = json.load(f)
        dataset = dataset_json["data"]
    with open("./result/bidaf_pred.json") as prediction_file:
        predictions = json.load(prediction_file)
    ret = evaluate(dataset, predictions)

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", default=100, type=int, help="hidden size")
    parser.add_argument("--dropout", default=0.0, type=float, help="dropout")
    parser.add_argument("--decay", default=0.999, type=float, help="exp moving average decay")
    parser.add_argument("--batch_size", default=32, type=int, help="batch_size")
    parser.add_argument("--max_seq_len", default=384, type=int, help="max context length")
    parser.add_argument("--max_query_len", default=64, type=int, help="max query length")
    parser.add_argument("--model_path", type=str, default="./save/bidaf_71", help="model path")
    parser.add_argument("--debug", action="store_true", help="debugging mode")
    args = parser.parse_args()
    ret = eval_qa(args)
    print(ret)
