from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from con_train.dataset import ContrastiveDataset, collate_fn
from con_train.model import ContrastiveModel


def read_data(fewrel, template):
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

        sent = ' '.join(tokens)
        text = template.format(sent=sent, head=head, tail=tail)
        rows.append(text)
    return rows


def get_contrastive_feature(sentences, tokenizer, max_length):
    contrastive_sentences = []
    for sentence in sentences:
        sentence_pair = [sentence, sentence]
        contrastive_sentences.append(sentence_pair)

    contrastive_features = []
    for sentence_pair in contrastive_sentences:
        sent_feature = tokenizer(
            sentence_pair,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors='pt'
        )
        contrastive_features.append(sent_feature)

    head_id = tokenizer.convert_tokens_to_ids('[E1]')  # 实体对应的id
    tail_id = tokenizer.convert_tokens_to_ids('[E2]')
    mask_id = tokenizer.mask_token_id
    entity_ids = []
    mask_ids = []
    for contrastive_feature in contrastive_features:
        en_ids = []
        mask = []
        for input_id in contrastive_feature['input_ids']:
            e1_ids = (input_id == head_id).nonzero().flatten().tolist()[0]
            e2_ids = (input_id == tail_id).nonzero().flatten().tolist()[0]
            en_ids.append((e1_ids, e2_ids))
            mask.append((input_id == mask_id).nonzero().flatten().tolist()[0])
        entity_ids.append(en_ids)
        mask_ids.append(mask)
    return contrastive_features, entity_ids, mask_ids


def train(fewrel, template, tokenizer, device, save_model, args):
    train_batch_size = 16
    train_epochs = 4
    max_length = args.max_len
    train_data = read_data(fewrel, template)
    contrastive_features, contrastive_entity_ids, mask_ids = get_contrastive_feature(train_data, tokenizer, max_length)

    dataset = ContrastiveDataset(contrastive_features, contrastive_entity_ids, mask_ids)
    model = ContrastiveModel(args.model_name, device, tokenizer)

    dataloader = DataLoader(dataset, shuffle=True, batch_size=train_batch_size, collate_fn=collate_fn)
    train_total = len(dataloader) * train_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.1},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=train_total)
    # Train
    model.to(device)
    model.zero_grad()

    for i in range(train_epochs):
        # print('======== Epoch {:} / {:} ========'.format(i + 1, train_epochs))
        # print('Training...')
        bar_desc = "Epoch %d of %d | Iteration" % (i + 1, train_epochs)
        epoch_iterator = tqdm(dataloader, desc=bar_desc)
        total_train_loss = 0

        for step, data in enumerate(epoch_iterator):
            model.train()
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            entity_ids = data['entity_ids'].to(device)
            mask_ids = data['mask_ids'].to(device)

            loss = model(
                input_ids,
                attention_mask=attention_mask,
                entity_ids=entity_ids,
                mask_ids=mask_ids,
                embedding=args.embedding
            )

            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scheduler.step()
            model.zero_grad()

        avg_train_loss = total_train_loss / len(dataloader)
        print("Average training loss: {}".format(avg_train_loss))
    torch.save(model, save_model)
