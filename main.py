from convokit import Corpus, download
corpus = Corpus(filename=download("conversations-gone-awry-corpus"))

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.utils import resample
import random

# Load data
df = corpus.get_utterances_dataframe()

# Prepend texts with author names in order to let model detect conversational patterns
df['text'] = df['speaker'] + " " + df['text']

# Only include rows which are not their section's header. Only include text and conversation_id columns
df = df.loc[df['meta.is_section_header'] != True, ['text','conversation_id']]

in_context_pairs = []
out_context_pairs = []

sample_size = 5000

# Group by conversation_id and create pairs of texts within each group
in_context_groups = df.groupby('conversation_id')
for group in in_context_groups:
    texts = [text for text in group[1]['text']]
    combos = list(combinations(texts, 2))
    in_context_pairs.extend(combo for combo in combos)

# Resample the pairs to get a sample of size sample_size
in_context_pairs = resample(in_context_pairs, replace=False, n_samples=sample_size, random_state=42)

# Create out-of-context pairs by swapping text_1 of one pair with text_2 of another pair
while len(out_context_pairs) < sample_size:
    i = random.randint(0, sample_size - 1)
    j = (i + sample_size // 2) % sample_size

    out_context_pairs.append((in_context_pairs[i][0], in_context_pairs[j][1]))

# Combine the in-context and out-of-context pairs into one dataframe and shuffle it
labeled = [(p[0], p[1], True) for p in in_context_pairs] + [(p[0], p[1], False) for p in out_context_pairs]
pairs = pd.DataFrame(labeled, columns=["text_1", "text_2", "in_context"])
shuffled = pairs.sample(frac=1)

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Load data
df = shuffled

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2)

# Tokenize data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

train_encodings = tokenizer(train_df['text_1'].tolist(), train_df['text_2'].tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_df['text_1'].tolist(), test_df['text_2'].tolist(), truncation=True, padding=True)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, train_df['in_context'].tolist())
test_dataset = CustomDataset(test_encodings, test_df['in_context'].tolist())

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
model.to('cuda')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3): # increase range for training
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    model.eval()
    correct_predictions = 0
    total_predictions = 0
    for batch in test_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += len(labels)

    accuracy = correct_predictions / total_predictions
    print(f'Epoch {epoch + 1} accuracy: {accuracy:.2f}')

text_1 = "I am going to the store."
text_2 = "I am going to the park."

inputs = tokenizer(text_1, text_2, return_tensors='pt', truncation=True)
outputs = model(**inputs.to('cuda'))
predictions = torch.argmax(outputs.logits).item()

torch.save(model, 'model.pt')

if predictions == 0:
    print("The two pieces of text are not in context.")
else:
    print("The two pieces of text are in context.")