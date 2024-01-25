import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from scan_utils.memory import MemoryBank
from scan_utils.scan_model import ScanModel, ScanDataset, ScanLoss


def find_k_nearest_neighbors(embeddings, k):
    from sklearn.neighbors import NearestNeighbors
    dist_func = 'euclidian'
    if dist_func == 'inner':
        tree = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric=lambda a, b: -(a@b))
    elif dist_func == 'euclidian':
        tree = NearestNeighbors(n_neighbors=k + 1)
    elif dist_func == 'cosine':
        tree = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric=lambda a, b: -((a@b) / (( (a@a) **.5) * ( (b@b) ** .5) )))
    tree.fit(embeddings)
    neighbors = tree.kneighbors(embeddings, k, return_distance=False)
    if dist_func == 'inner':
        result = []
        for i, neighbor in enumerate(neighbors):
            result.append(np.insert(neighbor, 0, i))
        neighbors = np.array(result)
    # print('neighbors', type(neighbors))
    # print('neighbors', neighbors.shape)
    # print('neighbors', neighbors[:50])

    return neighbors


def create_neighbor_dataset(memory_bank, embeddings, args):
    indices = memory_bank.mine_nearest_neighbors(args.topk, show_eval=False, calculate_accuracy=False)
    # indices = find_k_nearest_neighbors(embeddings, args.topk)
    examples = []
    for index in indices:
        anchor = index[0]
        neighbors = index[1:]
        for neighbor in neighbors:
            examples.append((anchor, neighbor))
    df = pd.DataFrame(examples, columns=["anchor", "neighbor"])
    return df


def get_predictions(model, dataloader):
    print("Getting predictions...")
    predictions, probabilities = [], []
    epoch_iterator = tqdm(dataloader, total=len(dataloader))
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(epoch_iterator):
            output_i = model(batch["anchor"])
            probabilities.extend(torch.nn.functional.softmax(output_i, dim=1))
            predictions.extend(torch.argmax(output_i, dim=1))
    return predictions, probabilities


def compute_scan_clustering(relation_embeddings, args, device):
    print("retrieving neighbors...")
    embeddings = torch.stack(relation_embeddings['embedding'].tolist())
    num_classes = args.n_rel
    memory_back = MemoryBank(embeddings, "", len(embeddings), embeddings.shape[-1], num_classes=num_classes)
    neighbor_dataset = create_neighbor_dataset(memory_back, embeddings, args)

    print("Computing SCAN clustering...")
    train_dataset = ScanDataset(neighbor_dataset, embeddings, mode="train")
    predict_dataset = ScanDataset(neighbor_dataset, embeddings, embeddings, mode="predict")
    model = ScanModel(num_classes, args.dropout, embeddings.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-8)
    criterion = ScanLoss()
    criterion.to(device)

    # get dataloader
    batch_size = args.scan_batch_size
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=train_dataset.collate_fn, batch_size=batch_size)
    predict_dataloader = DataLoader(predict_dataset, shuffle=False, collate_fn=predict_dataset.collate_fn_predict, batch_size=batch_size)

    # train
    train_iterator = range(int(args.num_epochs))
    for epoch in train_iterator:
        bar_desc = "Epoch %d of %d | num classes %d | Iteration" % (epoch + 1, len(train_iterator), num_classes)
        epoch_iterator = tqdm(train_dataloader, desc=bar_desc)
        for step, batch in enumerate(epoch_iterator):
            optimizer.zero_grad()
            model.zero_grad()
            anchor, neighbor = batch["anchor"], batch["neighbor"]
            anchors_output, neighbors_output = model(anchor), model(neighbor)
            total_loss, consistency_loss, entropy_loss = criterion(anchors_output, neighbors_output)
            total_loss.backward()
            optimizer.step()
    optimizer.zero_grad()
    model.zero_grad()

    # predict
    predictions, probabilities = get_predictions(model, predict_dataloader)
    return predictions


