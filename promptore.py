"""PromptORE

---
PromptORE
Copyright (C) 2022-2023 Alteca.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import argparse
import random
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import evaluate_promptore, compute_kmeans_clustering, estimate_n_rel, parse_fewrel
from con_train.train import train
from scan_utils.scan_execute import compute_scan_clustering


def tokenize(tokenizer, text: str, max_len: int) -> tuple:
    """Tokenize input text
    Args:
        tokenizer (any): BertTokenizer (or RobertaTokenizer)
        text (str): text to tokenize
        max_len (int): max nb of tokens

    Returns:
        tuple: input ids and attention masks
    """
    encoded_dict = tokenizer.encode_plus(
        text,                           # Sentence to encode.
        add_special_tokens=True,        # Add '[CLS]' and '[SEP]'
        max_length=max_len,        # Pad & truncate all sentences.
        padding='max_length',
        truncation=True,
        return_attention_mask=True,     # Construct attn. masks.
        return_tensors='pt',            # Return pytorch tensors.
    )
    input_ids = encoded_dict['input_ids'].view(-1)
    attention_mask = encoded_dict['attention_mask'].view(-1)
    return input_ids, attention_mask


def data_process(fewrel, template, tokenizer, max_len, mask_id):
    # Tokenize fewrel
    rows = []
    for index, instance in tqdm(fewrel.iterrows(), total=len(fewrel)):
        sentence = instance['tokens'].copy()

        tokens = []  # 句子中的所有 tokens，添加实体标记后，即下边4个
        head_start_mark = "[E1]"
        head_end_mark = "[/E1]"
        tail_start_mark = "[E2]"
        tail_end_mark = "[/E2]"
        for i, token in enumerate(sentence):
            if i == instance['h_start']:
                tokens.append(head_start_mark)
            if i == instance['h_end']+1:
                tokens.append(head_end_mark)
            if i == instance['t_start']:
                tokens.append(tail_start_mark)
            if i == instance['t_end']+1:
                tokens.append(tail_end_mark)
            tokens.append(token)

        head = ' '.join(sentence[instance['h_start']:instance['h_end']+1])
        tail = ' '.join(sentence[instance['t_start']:instance['t_end']+1])
        head_ids = tokenizer(' '+head_start_mark, add_special_tokens=False)['input_ids']  # 实体对应的id
        tail_ids = tokenizer(' '+tail_start_mark, add_special_tokens=False)['input_ids']
        # print(head_ids, tail_ids)
        # assert 1==2

        sent = ' '.join(tokens)
        text = template.format(sent=sent, head=head, tail=tail)
        if index == 0:
            print('Example of text: ', text)

        input_ids, attention_mask = tokenize(tokenizer, text, max_len)

        head_start = head_end = tail_start = tail_end = -1
        for i in range(len(input_ids)):
            if head_start == -1 and input_ids[i:i+len(head_ids)].numpy().tolist() == head_ids:
                head_start, head_end = i, i + len(head_ids)
            if tail_start == -1 and input_ids[i:i+len(tail_ids)].numpy().tolist() == tail_ids:
                tail_start, tail_end = i, i + len(tail_ids)
        assert head_start != -1 and tail_start != -1
        entity_ids = [head_start, head_end, tail_start, tail_end]

        rows.append({
            'input_tokens': input_ids,
            'input_attention_mask': attention_mask,
            'input_mask': (input_ids == mask_id).nonzero().flatten().item(),
            'output_r': instance['r'],
            'entity_ids': entity_ids,
        })
    return rows


def compute_relation_embedding(fewrel: pd.DataFrame, template: str = '{e1} [MASK] {e2}.', max_len=128,
                               device: str = 'cuda', args=None) -> pd.DataFrame:
    """Compute relation embedding for the dataframe

    Args:
        fewrel (pd.DataFrame): fewrel dataset
        template (str, optional): template to use.
            Authorized parameters are {e1} {e2} {sent}. Defaults to '{e1} [MASK] {e2}.'.
        max_len (int, optional): max nb of tokens. Defaults to 128.
        device (str, optional): Pytorch device to use. Defaults to cuda
        args (Any, optional): args. Defaults to None.

    Returns:
        pd.DataFrame: fewrel dataset with relation embeddings
    """
    print('Computing relation embeddings...')
    fewrel = fewrel.copy()
    tokenizer = BertTokenizer.from_pretrained(args.model_name, do_lower_case=True)
    model = AutoModel.from_pretrained(args.model_name, output_attentions=False)
    # args.model_name = 'roberta-large'
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=True)
    mask_id = tokenizer.mask_token_id
    entity_list = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
    tokenizer.add_special_tokens({'additional_special_tokens': entity_list})
    model.resize_token_embeddings(len(tokenizer))

    save_model = os.path.join('output/checkpoint.pt')
    # train(fewrel, template, tokenizer, device, save_model, args)

    rows = data_process(fewrel, template, tokenizer, max_len, mask_id)
    complete_fewrel = pd.DataFrame(rows)
    complete_fewrel['output_label'] = pd.factorize(complete_fewrel['output_r'])[0]

    # Predict embeddings
    # model = torch.load(save_model)
    model.to(device)
    model.eval()

    tokens = torch.stack(complete_fewrel['input_tokens'].tolist(), dim=0)  # 在维度上连接（concatenate）若干个张量
    attention_mask = torch.stack(complete_fewrel['input_attention_mask'].tolist(), dim=0)
    masks = torch.Tensor(complete_fewrel['input_mask'].tolist()).long()
    entity_ids = torch.Tensor(complete_fewrel['entity_ids'].tolist()).long()
    dataset = TensorDataset(tokens, attention_mask, masks, entity_ids)
    dataloader = DataLoader(dataset, num_workers=1, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        embeddings = []
        for batch in tqdm(dataloader):
            tokens, attention_mask, mask, entity_ids = batch
            tokens = tokens.to(device)
            attention_mask = attention_mask.to(device)
            result = model(tokens, attention_mask, output_hidden_states=True)
            # result = model.encode(tokens, attention_mask=attention_mask)
            output_embedding = result.hidden_states[-1]
            batch_size = output_embedding.shape[0]

            out = result[0].detach()
            mask_embedding = out[torch.arange(batch_size), mask]

            head_embedding = []
            tail_embedding = []
            for i in range(batch_size):
                head_embedding.append(torch.mean(output_embedding[i, entity_ids[i, 0]:entity_ids[i, 1]], dim=0))
                tail_embedding.append(torch.mean(output_embedding[i, entity_ids[i, 2]:entity_ids[i, 3]], dim=0))
            head_embedding = torch.stack(head_embedding)
            tail_embedding = torch.stack(tail_embedding)
            entity_embedding = torch.cat([head_embedding, tail_embedding], dim=1)

            if args.embedding == 'mask':
                embedding = mask_embedding
            elif args.embedding == 'entity':
                embedding = entity_embedding
            elif args.embedding == 'entity_mask':
                embedding = torch.cat([entity_embedding, mask_embedding], dim=1)
            embeddings.append(embedding)
            del result
        embeddings = torch.cat(embeddings, dim=0).detach().to('cpu')
        embeddings_list = list(embeddings)

    model.to('cpu')
    complete_fewrel['embedding'] = embeddings_list
    return complete_fewrel


if __name__ == "__main__":
    # python3 promptore.py --seed=0 --n-rel=80 --max-len=150 --files "data/train_wiki.json" "data/val_wiki.json"
    # python3 promptore.py --seed=0 --n-rel=25 --max-len=500 --files "data/val_nyt.json"
    # python3 promptore.py --seed=0 --n-rel=10 --max-len=250 --files "data/val_pubmed.json"

    parser = argparse.ArgumentParser(prog='PromptORE')
    parser.add_argument('--seed', default=0, help='Random state', type=int, required=True)
    parser.add_argument('--n-rel', help='Number of clusters', type=int)
    parser.add_argument('--auto-n-rel', help='Wether to estimate the number of clusters', action='store_true')
    parser.add_argument('--min-n-rel', help='In case of cluster estimation, min nb of cluster', type=int, default=10)
    parser.add_argument('--max-n-rel', help='In case of cluster estimation, max nb of cluster', type=int, default=300)
    parser.add_argument('--step-n-rel', help='In case of cluster estimation, step', type=int, default=5)
    parser.add_argument('--files', help='File(s) to load from Fewrel', default=[], nargs='+')
    parser.add_argument('--max-len', help='Maximum number of tokens (fewrel=150, fewrel_nyt=500, fewrel_pubmed=250)',
                        type=int, required=True)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=256)
    parser.add_argument('--model_name', help='Pre-trained Language Models', type=str, default='bert-base-uncased')
    parser.add_argument('--embedding', type=str, default='entity_mask',
                        help='Relation embeddings to use, [entity_mask, entity, mask]')
    parser.add_argument('--topk', type=int, default=10, help='numbers of neighbors retrieved to build SCAN training set')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout for scan model')
    parser.add_argument('--scan_batch_size', type=int, default=128, help='batch size for scan model')
    parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs for scan model')
    parser.set_defaults(auto_n_rel=False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Read docred files
    files = args.files
    fewrel_files = [parse_fewrel(file) for file in files]
    datasets = pd.concat(fewrel_files).reset_index(drop=True)   # 将多个文件合并成一个文件

    # template = "{sent} [SEP] The relation between {head} and {tail} is [MASK]"
    template = "{sent} [SEP] {head} [MASK] {tail}"
    data_output = compute_relation_embedding(
        fewrel=datasets, template=template, max_len=args.max_len, device=device, args=args)

    # Compute clustering
    if args.auto_n_rel:
        n_rel = estimate_n_rel(data_output, args.seed, (args.min_n_rel, args.max_n_rel), args.step_n_rel)
        print(f'Estimated n_rel={n_rel}')
    else:
        n_rel = args.n_rel

    predicted_labels = compute_scan_clustering(data_output, args=args, device=device)
    # predicted_labels = compute_kmeans_clustering(data_output, n_rel, args.seed)

    # Evaluation
    b3, b3_prec, b3_rec, v, v_hom, v_comp, ari = evaluate_promptore(data_output, predicted_labels)
    print(f'B3:        prec={b3_prec:.4f} rec={b3_rec:.4f} f1={b3:.4f}')
    print(f'V-measure: hom={v_hom:.4f} comp={v_comp:.4f} f1={v:.4f}')
    print(f'           ARI={ari:.4f}')
