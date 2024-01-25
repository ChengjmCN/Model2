import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.kernel_ridge import KernelRidge
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics.cluster import contingency_matrix, adjusted_rand_score, homogeneity_score, completeness_score


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


def evaluate_promptore(embeddings: pd.DataFrame, predicted_labels) -> tuple:
    """Evaluate PromptORE
    Args:
        embeddings (pd.DataFrame): fewrel
        predicted_labels (torch.Tensor): predicted labels

    Returns:
        tuple: scores
    """
    labels = torch.Tensor(embeddings['output_label'].tolist()).long()
    predicted_labels = torch.Tensor(predicted_labels).long()
    # print("labels.shape: ", labels.shape)
    # print("predicted_labels.shape: ", predicted_labels.shape)

    ari = adjusted_rand_score(labels, predicted_labels)
    v, v_hom, v_comp = v_measure(labels, predicted_labels)
    b3, b3_prec, b3_rec = bcubed(labels, predicted_labels)

    return b3, b3_prec, b3_rec, v, v_hom, v_comp, ari


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


def compute_kmeans_clustering(fewrel_relation_embeddings: pd.DataFrame, n_rel: int, random_state: int):
    """Compute kmeans clustering with fixed nb of clusters
    Args:
        fewrel_relation_embeddings (pd.DataFrame): relation embeddings
        n_rel (int): number of relations (nb of clusters)
        random_state (int): random state
    Returns:
        torch.Tensor: predicted labels
    """
    embeddings = torch.stack(fewrel_relation_embeddings['embedding'].tolist())

    model = KMeans(init='k-means++', n_init=10, n_clusters=n_rel, random_state=random_state)
    model.fit(embeddings)
    predicted_labels = model.predict(embeddings)

    return predicted_labels


def estimate_n_rel(fewrel_relation_embeddings: pd.DataFrame, random_state: int, k_range: tuple = [10, 300],
                   k_step: int = 5) -> int:
    """Estimate number of clusters using the elbow rule

    Args:
        fewrel_relation_embeddings (pd.DataFrame): relation embeddings
        k_range (tuple, optional): range of clusters to test. Defaults to [10, 300].
        k_step (int, optional): step. Defaults to 5.
        random_state (int): random state
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


def compute_sbert_embedding(fewrel: pd.DataFrame, args):
    print('Computing sentence bert embeddings...')
    from sentence_transformers import SentenceTransformer
    fewrel = fewrel.copy()
    rows = []
    for _, instance in tqdm(fewrel.iterrows(), total=len(fewrel)):
        tokens = instance['tokens'].copy()
        sentence = ' '.join(tokens)
        rows.append({'sentence': sentence, 'output_r': instance['r']})
    complete_fewrel = pd.DataFrame(rows)
    complete_fewrel['output_label'] = torch.Tensor(pd.factorize(complete_fewrel['output_r'])[0])
    tokenizer = SentenceTransformer('bert-base-nli-mean-tokens')
    tokenizer.max_seq_length = args.max_len
    embeddings = tokenizer.encode(complete_fewrel['sentence'], batch_size=args.batch_size, show_progress_bar=True)
    embeddings = torch.from_numpy(embeddings)

    complete_fewrel['embedding'] = list(embeddings)
    return complete_fewrel


def init_word_weights(model, tokenizer):
    """
    在提示模板中使用虚拟实体类型， 类似KnowPrompt
    """
    entity_type_list = ["[sub]", "[obj]"]
    tokenizer.add_special_tokens({'additional_special_tokens': entity_type_list})

    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        word_embeddings = model.get_input_embeddings()
        entity_type_list = ["[sub]", "[obj]"]
        entity_type_list_ids = [a[0] for a in tokenizer(entity_type_list, add_special_tokens=False)['input_ids']]
        meaning_word = [a[0] for a in tokenizer(["person", "organization", "location", "date", "country"],
                                                add_special_tokens=False)['input_ids']]
        for i, idx in enumerate(entity_type_list_ids):
            word_embeddings.weight[idx] = torch.mean(word_embeddings.weight[meaning_word], dim=0)
