import torch
import random
from torch.utils.data import Dataset
import torch.nn.functional as functional


class ScanModel(torch.nn.Module):
    def __init__(self, num_labels, dropout, hidden_dim=768):
        super(ScanModel, self).__init__()
        self.num_labels = num_labels
        self.classifier = torch.nn.Linear(hidden_dim, num_labels)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dropout = dropout

    def forward(self, feature):
        if self.dropout is not None:
            dropout = torch.nn.Dropout(p=self.dropout)
            feature = dropout(feature)
        output = self.classifier(feature)
        return output


class ScanDataset(Dataset):
    def __init__(self, neighbor_df, embeddings, test_embeddings="", mode="train"):
        self.neighbor_df = neighbor_df
        self.embeddings = embeddings
        self.mode = mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if mode == "train":
            self.examples = self.load_data()
        elif mode == "predict":
            self.examples = test_embeddings

    def load_data(self):
        examples = []
        for i, j in zip(self.neighbor_df["anchor"], self.neighbor_df["neighbor"]):
            examples.append((i, j))
        random.shuffle(examples)
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        sample = {}
        if self.mode == "train":
            anchor, neighbor = self.examples[item]
            sample = {"anchor": anchor, "neighbor": neighbor}
        elif self.mode == "predict":
            anchor = self.examples[item]
            sample = {"anchor": anchor}
        return sample

    def collate_fn(self, batch):
        anchors = torch.tensor([i["anchor"] for i in batch])
        out = self.embeddings[anchors].to(self.device)
        neighbors = torch.tensor([i["neighbor"] for i in batch])
        out_2 = self.embeddings[neighbors].to(self.device)
        return {"anchor": out, "neighbor": out_2}

    def collate_fn_predict(self, batch):
        out = torch.vstack([i["anchor"] for i in batch]).to(self.device)
        return {"anchor": out}


def entropy(x, input_as_probabilities):
    """
    Helper function to compute the entropy over the batch

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    EPS = 1e-8
    if input_as_probabilities:
        x_ = torch.clamp(x, min=EPS)
        b = x_ * torch.log(x_)
    else:
        b = functional.softmax(x, dim=1) * functional.log_softmax(x, dim=1)

    if len(b.size()) == 2:  # Sample-wise entropy
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


class ScanLoss(torch.nn.Module):
    def __init__(self, entropy_weight=2, entropy="entropy"):
        super(ScanLoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.bce = torch.nn.BCELoss()
        self.entropy_weight = entropy_weight  # Default = 2.0
        self.entropy = entropy

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)

        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)

        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities=True)

        # Total loss
        # print (consistency_loss, entropy_loss)
        total_loss = consistency_loss - self.entropy_weight * entropy_loss

        return total_loss, consistency_loss, entropy_loss
