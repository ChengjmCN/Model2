import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig


class ContrastiveModel(nn.Module):
    def __init__(self, model_name, device, tokenizer, dropout=0.1, temp=0.05):
        super(ContrastiveModel, self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.attention_probs_dropout_prob = dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = dropout
        self.model = AutoModel.from_pretrained(model_name, config=config)
        if tokenizer is not None:
            self.model.resize_token_embeddings(len(tokenizer))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        self.device = device

    def forward(self, input_ids, attention_mask, entity_ids, mask_ids, embedding):
        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
        outputs = self.model(input_ids, attention_mask, output_hidden_states=True)

        last_hidden = outputs[0]  # last_hidden [bs, sent_length, hidden]
        last_hidden = self.dense(last_hidden)
        last_hidden = self.activation(last_hidden)
        sent_length = last_hidden.size(1)
        last_hidden = last_hidden.view((batch_size, num_sent, sent_length, last_hidden.size(-1)))
        sent1_hidden, sent2_hidden = last_hidden[:, 0], last_hidden[:, 1]  # [bs, sent_length, hidden]

        z1_entity = []
        z2_entity = []
        z1_mask = []
        z2_mask = []
        z1 = []
        z2 = []
        for i in range(len(entity_ids)):
            sent1_head_ids, sent1_tail_ids = entity_ids[i][0][0], entity_ids[i][0][1]
            sent2_head_ids, sent2_tail_ids = entity_ids[i][1][0], entity_ids[i][1][1]
            sent1_mask_ids, sent2_mask_ids = mask_ids[i][0], mask_ids[i][1]
            # sent1
            sent1_head_entity = sent1_hidden[i][sent1_head_ids]
            sent1_tail_entity = sent1_hidden[i][sent1_tail_ids]
            sent1_mask = sent1_hidden[i][sent1_mask_ids]
            # sent2
            sent2_head_entity = sent2_hidden[i][sent2_head_ids]
            sent2_tail_entity = sent2_hidden[i][sent2_tail_ids]
            sent2_mask = sent2_hidden[i][sent2_mask_ids]

            sent1_relation_expresentation = torch.cat([sent1_head_entity, sent1_tail_entity], dim=-1)
            sent2_relation_expresentation = torch.cat([sent2_head_entity, sent2_tail_entity], dim=-1)
            z1_entity.append(sent1_relation_expresentation.unsqueeze(0))
            z2_entity.append(sent2_relation_expresentation.unsqueeze(0))
            z1_mask.append(sent1_mask.unsqueeze(0))
            z2_mask.append(sent2_mask.unsqueeze(0))

            if embedding == 'mask':
                z1.append(sent1_mask.unsqueeze(0))
                z2.append(sent2_mask.unsqueeze(0))
            elif embedding == 'entity':
                z1.append(sent1_relation_expresentation.unsqueeze(0))
                z2.append(sent2_relation_expresentation.unsqueeze(0))
            elif embedding == 'entity_mask':
                z1.append(torch.cat([sent1_relation_expresentation, sent1_mask], dim=-1).unsqueeze(0))
                z2.append(torch.cat([sent2_relation_expresentation, sent2_mask], dim=-1).unsqueeze(0))

        z1 = torch.cat(z1_entity, dim=0)
        z2 = torch.cat(z2_entity, dim=0)

        cos_sim = self.cos(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temp
        con_labels = torch.arange(cos_sim.size(0)).long().to(self.device)
        loss_fct = nn.CrossEntropyLoss()
        cl_loss = loss_fct(cos_sim, con_labels)

        # print('cos_sim', cos_sim)
        # print('con_labels', con_labels)
        # print('cl_loss', cl_loss)

        return cl_loss

    def encode(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask, output_hidden_states=True)
        return outputs
