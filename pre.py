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
import json
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.kernel_ridge import KernelRidge
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix, adjusted_rand_score, homogeneity_score, completeness_score
from yellowbrick.cluster import KElbowVisualizer
from tqdm.auto import tqdm


################################################################################
# Metrics
################################################################################


def bcubed(targets, predictions, beta: float = 1):
    """B3 metric (see Baldwin1998)
    Args:
        targets (torch.Tensor): true labels
        predictions (torch.Tensor): predicted labels
        beta (float, optional): beta for f_score. Defaults to 1.
    Returns:
        Tuple[float, float, float]: b3 f1, precision and recall
    """

    cont_mat = contingency_matrix(targets, predictions)
    cont_mat_norm = cont_mat / cont_mat.sum()

    precision = np.sum(cont_mat_norm * (cont_mat /
                       cont_mat.sum(axis=0))).item()
    recall = np.sum(cont_mat_norm * (cont_mat /
                    np.expand_dims(cont_mat.sum(axis=1), 1))).item()
    f1_score = (1 + beta) * precision * recall / (beta * (precision + recall))

    return f1_score, precision, recall


def v_measure(targets, predictions):
    """V-measure
    Args:
        targets (torch.Tensor): true labels
        predictions (torch.Tensor): predictions
    Returns:
        Tuple[float, float, float]: V-measure f1, homogeneity (~prec), completeness (~rec)
    """
    homogeneity = homogeneity_score(targets, predictions)
    completeness = completeness_score(targets, predictions)
    v = 2 * homogeneity * completeness / (homogeneity + completeness)

    return v, homogeneity, completeness


def evaluate_promptore(fewrel: pd.DataFrame, predicted_labels: torch.Tensor) -> tuple:
    """Evaluate PromptORE
    Args:
        fewrel (pd.DataFrame): fewrel
        predicted_labels (torch.Tensor): predicted labels

    Returns:
        tuple: scores
    """
    labels = torch.Tensor(fewrel['output_label'].tolist()).long()

    ari = adjusted_rand_score(labels, predicted_labels)
    v, v_hom, v_comp = v_measure(labels, predicted_labels)
    b3, b3_prec, b3_rec = bcubed(labels, predicted_labels)

    return b3, b3_prec, b3_rec, v, v_hom, v_comp, ari

################################################################################
# Dataset
################################################################################


def parse_fewrel(path: str, expand: bool = False) -> pd.DataFrame:
    """Parse fewrel dataset. Dataset can be downloaded at:
        https://github.com/thunlp/FewRel/tree/master/data
    Args:
        path (str): path to json fewrel file
        expand (bool, Optional): to expand every instance (entity mentionned twice in
                                sentence -> 2 instances). Defaults to False.
    Returns:
        pd.DataFrame: parsed fewrel dataset
    """
    with open(path, 'r', encoding='utf-8') as file:
        fewrel_json = json.load(file)

    fewrel_tuples = []
    for relation, instances in fewrel_json.items():
        for instance in instances:
            for i_h, h_pos in enumerate(instance['h'][2]):
                for i_t, t_pos in enumerate(instance['t'][2]):
                    fewrel_tuples.append({
                        'tokens': instance['tokens'],
                        'r': relation,
                        'h': instance['h'][0],
                        'h_id': instance['h'][1],
                        'h_count': i_h,
                        'h_start': h_pos[0],
                        'h_end': h_pos[len(h_pos) - 1],
                        't': instance['t'][0],
                        't_id': instance['t'][1],
                        't_count': i_t,
                        't_start': t_pos[0],
                        't_end': t_pos[len(t_pos) - 1],
                    })
                    if not expand:
                        break
                if not expand:
                    break
    return pd.DataFrame(fewrel_tuples)

################################################################################
# PromptORE
################################################################################


def to_device(data, device):
    """Move data to device
    Args:
        data (Any): data
        device (str): device
    Returns:
        Any: moved data
    """
    def to_device_dict_(k: str, v, device: str):
        """Util function to move a dict (ignore variables ending with '_')
        Args:
            k (str): key
            v (Any): value
            device (str): device
        Returns:
            Any: moved value
        """
        if k.endswith('_'):
            return v
        else:
            return to_device(v, device)

    if isinstance(data, tuple):
        data = (to_device(e, device) for e in data)
    elif isinstance(data, list):
        data = [to_device(e, device) for e in data]
    elif isinstance(data, dict):
        data = {k: to_device_dict_(k, v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.to(device)

    return data


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


def init_word_weights(model, tokenizer):
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        word_embeddings = model.get_input_embeddings()
        entity_type_list = ["[sub]", "[obj]"]
        entity_type_list_ids = [a[0] for a in tokenizer(entity_type_list, add_special_tokens=False)['input_ids']]
        meaning_word = [a[0] for a in tokenizer(["person", "organization", "location", "date", "country"],
                                                add_special_tokens=False)['input_ids']]
        for i, idx in enumerate(entity_type_list_ids):
            word_embeddings.weight[idx] = torch.mean(word_embeddings.weight[meaning_word], dim=0)


def compute_promptore_relation_embedding(fewrel: pd.DataFrame, template: str = '{e1} [MASK] {e2}.',
                                         max_len=128, device: str = 'cuda', args=None) -> pd.DataFrame:
    """Compute PromptORE relation embedding for the dataframe

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
    fewrel = fewrel.copy()
    # Setup tokenizer + bert
    tokenizer = BertTokenizer.from_pretrained(args.model_name, do_lower_case=True)
    mask_id = tokenizer.mask_token_id
    bert = BertModel.from_pretrained(args.model_name, output_attentions=False)

    entity_list = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
    entity_type_list = ["[sub]", "[obj]"]
    tokenizer.add_special_tokens({'additional_special_tokens': entity_list})
    tokenizer.add_special_tokens({'additional_special_tokens': entity_type_list})
    # bert.resize_token_embeddings(len(tokenizer))

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
    complete_fewrel = pd.DataFrame(rows)
    complete_fewrel['output_label'] = pd.factorize(complete_fewrel['output_r'])[0]

    # Predict embeddings
    bert.to(device)
    bert.eval()

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
            result = bert(tokens, attention_mask, output_hidden_states=True)
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

    bert.to('cpu')
    complete_fewrel['embedding'] = embeddings_list
    return complete_fewrel


def compute_kmeans_clustering(fewrel_relation_embeddings: pd.DataFrame, n_rel: int, random_state: int):
    """Compute kmeans clustering with fixed nb of clusters
    Args:
        fewrel_relation_embeddings (pd.DataFrame): relation embeddings
        n_rel (int): number of relations (nb of clusters)
    Returns:
        torch.Tensor: predicted labels
    """
    embeddings = torch.stack(fewrel_relation_embeddings['embedding'].tolist())

    model = KMeans(init='k-means++', n_init=10, n_clusters=n_rel, random_state=random_state)
    predicted_labels = model.fit(embeddings)
    predicted_labels = model.predict(embeddings)

    return predicted_labels


def estimate_n_rel(fewrel_relation_embeddings: pd.DataFrame, random_state: int, k_range: tuple = [10, 300],
                   k_step: int = 5) -> int:
    """Estimate number of clusters using the elbow rule

    Args:
        fewrel_relation_embeddings (pd.DataFrame): relation embeddings
        k_range (tuple, optional): range of clusters to test. Defaults to [10, 300].
        k_step (int, optional): step. Defaults to 5.

    Returns:
        int: estimated number of clusters
    """
    embeddings = torch.stack(fewrel_relation_embeddings['embedding'].tolist())

    ks = np.arange(k_range[0], k_range[1], k_step)
    model = KMeans(init='k-means++', n_init=10, random_state=random_state)
    visualizer = KElbowVisualizer(
        model, k=ks, metric='silhouette', timings=False, locate_elbow=False)
    visualizer.fit(embeddings)
    silhouette = pd.DataFrame()
    silhouette['ks'] = ks
    silhouette['scores'] = visualizer.k_scores_

    # Kernel ridge
    model = KernelRidge(kernel='rbf', degree=3, gamma=1e-3)
    X = silhouette['ks'].values.reshape(-1, 1)
    model.fit(X=X, y=silhouette['scores'])
    p = model.predict(X=X)

    k_elbow = silhouette['ks'][p.argmax()]
    return k_elbow


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

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
    parser.set_defaults(auto_n_rel=False)
    args = parser.parse_args()

    # Read docred files
    files = args.files
    fewrel_files = [parse_fewrel(file) for file in files]
    fewrel = pd.concat(fewrel_files).reset_index(drop=True)   # 将多个文件合并成一个文件

    # template = "{sent} [SEP] [sub] {head} [sub] [MASK] [obj] {tail} [obj]"
    template = "{sent} [SEP] {head} [MASK] {tail}"

# Compute relation embeddings
    relation_embeddings = compute_promptore_relation_embedding(
        fewrel=fewrel, template=template, max_len=args.max_len, device=device, args=args)
        # fewrel=fewrel, template="{sent}. The relation between {e1} and {e2} is [MASK]", max_len=args.max_len, device=device, args=args)

    # Compute clustering
    if args.auto_n_rel:
        n_rel = estimate_n_rel(relation_embeddings, args.seed, (args.min_n_rel, args.max_n_rel), args.step_n_rel)
        print(f'Estimated n_rel={n_rel}')
    else:
        n_rel = args.n_rel

    predicted_labels = compute_kmeans_clustering(relation_embeddings, n_rel, args.seed)

    # Evaluation
    b3, b3_prec, b3_rec, v, v_hom, v_comp, ari = evaluate_promptore(relation_embeddings, predicted_labels)
    print(f'B3:        prec={b3_prec:.4f} rec={b3_rec:.4f} f1={b3:.4f}')
    print(f'V-measure: hom={v_hom:.4f} comp={v_comp:.4f} f1={v:.4f}')
    print(f'           ARI={ari:.4f}')
    # print(f'B3: prec={b3_prec} rec={b3_rec} f1={b3}')
    # print(f'V-measure: hom={v_hom} comp={v_comp} f1={v}')
    # print(f'ARI={ari}')
