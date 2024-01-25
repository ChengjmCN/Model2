import torch
from torch.utils.data import Dataset


class ContrastiveDataset(Dataset):
    def __init__(self, contrastive_features, contrastive_entity_ids, mask_ids):
        super(ContrastiveDataset, self).__init__()
        self.contrastive_features = contrastive_features
        self.contrastive_entity_ids = contrastive_entity_ids
        self.mask_ids = mask_ids

    def __getitem__(self, ids):
        contrastive_features = self.contrastive_features
        contrastive_entity_ids = self.contrastive_entity_ids
        mask_ids = self.mask_ids
        output = {
            "input_ids": contrastive_features[ids]["input_ids"],
            "attention_mask": contrastive_features[ids]["attention_mask"],
            "entity_ids": contrastive_entity_ids[ids],
            "mask_ids": mask_ids[ids]
        }

        return output

    def __len__(self):
        return len(self.contrastive_features)


def collate_fn(batch):
    batch_input_ids = [data["input_ids"].unsqueeze(0) for data in batch]
    batch_attention_mask = [data["attention_mask"].unsqueeze(0) for data in batch]
    batch_entity_ids = [data["entity_ids"] for data in batch]
    batch_mask_ids = [data["mask_ids"] for data in batch]

    batch_input_ids = torch.cat(batch_input_ids, dim=0)
    batch_attention_mask = torch.cat(batch_attention_mask, dim=0)
    batch_entity_ids = torch.tensor(batch_entity_ids)
    batch_mask_ids = torch.tensor(batch_mask_ids)

    output = {
        'input_ids': batch_input_ids,
        'attention_mask': batch_attention_mask,
        'entity_ids': batch_entity_ids,
        'mask_ids': batch_mask_ids
    }

    return output

