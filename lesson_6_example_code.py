# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 14:55:58 2025

@author: taske
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def download_conll_data():
    try:
        dataset = load_dataset("conll2003", trust_remote_code=True)
        train_data = dataset["train"]
        label_list = train_data.features["ner_tags"].feature.names
        sentences = []

        for example in train_data:
            tokens = example["tokens"]
            ner_tags = example["ner_tags"]
            sentence = [(token, label_list[tag]) for token, tag in zip(tokens, ner_tags)]
            sentences.append(sentence)

        return sentences
    except Exception as e:
        print("Falling back to sample data...")
        return fallback_sample_data()

def fallback_sample_data():
    sample_data = [
        ("EU", "B-ORG"), ("rejects", "O"), ("German", "B-MISC"),
        ("call", "O"), ("to", "O"), ("boycott", "O"), ("British", "B-MISC"),
        ("lamb", "O"), (".", "O"), ("Peter", "B-PER"), ("Blackburn", "I-PER"),
    ]
    sentences, current = [], []
    for word, tag in sample_data:
        current.append((word, tag))
        if word == ".":
            sentences.append(current)
            current = []
    return sentences

class NERDataset(Dataset):
    def __init__(self, sentences, word_to_idx, tag_to_idx, max_length=50):
        self.sentences = sentences
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        word_indices = []
        tag_indices = []
        for word, tag in sentence:
            word_idx = self.word_to_idx.get(word.lower(), self.word_to_idx['<UNK>'])
            tag_idx = self.tag_to_idx[tag]
            word_indices.append(word_idx)
            tag_indices.append(tag_idx)

        while len(word_indices) < self.max_length:
            word_indices.append(self.word_to_idx['<PAD>'])
            tag_indices.append(self.tag_to_idx['O'])

        word_indices = word_indices[:self.max_length]
        tag_indices = tag_indices[:self.max_length]

        return torch.tensor(word_indices), torch.tensor(tag_indices)

class NERModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        super(NERModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(hidden_dim, num_tags)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = self.relu(self.hidden(x))
        x = self.dropout(x)
        return self.output(x)

def train_ner_model(train_fraction=1.0):
    sentences = download_conll_data()

    # Reduce dataset size by fraction if needed
    if train_fraction < 1.0:
        cut = int(len(sentences) * train_fraction)
        sentences = sentences[:cut]
        print(f"Training on {cut} sentences ({train_fraction*100:.1f}%)")

    word_counter = Counter()
    tag_counter = Counter()
    for s in sentences:
        for w, t in s:
            word_counter[w.lower()] += 1
            tag_counter[t] += 1

    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word in word_counter:
        word_to_idx[word] = len(word_to_idx)

    tag_to_idx = {tag: idx for idx, tag in enumerate(tag_counter)}
    idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}

    train_size = int(0.8 * len(sentences))
    train_dataset = NERDataset(sentences[:train_size], word_to_idx, tag_to_idx)
    val_dataset = NERDataset(sentences[train_size:], word_to_idx, tag_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    model = NERModel(len(word_to_idx), 100, 128, len(tag_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tag_to_idx['O'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_accuracies = [], []
    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch_words, batch_tags in train_loader:
            batch_words, batch_tags = batch_words.to(device), batch_tags.to(device)
            optimizer.zero_grad()
            outputs = model(batch_words)
            loss = criterion(outputs.view(-1, len(tag_to_idx)), batch_tags.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_words, batch_tags in val_loader:
                batch_words, batch_tags = batch_words.to(device), batch_tags.to(device)
                outputs = model(batch_words)
                preds = torch.argmax(outputs, dim=2)
                mask = batch_tags != tag_to_idx['O']
                correct += ((preds == batch_tags) & mask).sum().item()
                total += mask.sum().item()
        acc = correct / total if total > 0 else 0
        train_losses.append(total_loss / len(train_loader))
        val_accuracies.append(acc)
        print(f"Epoch {epoch:02d}, Loss: {total_loss:.4f}, Val Acc: {acc:.4f}")

    return model, word_to_idx, tag_to_idx, idx_to_tag, train_losses, val_accuracies

def predict_ner(model, sentence, word_to_idx, idx_to_tag):
    model.eval()
    words = sentence.lower().split()
    indices = [word_to_idx.get(w, word_to_idx['<UNK>']) for w in words]
    while len(indices) < 50:
        indices.append(word_to_idx['<PAD>'])
    input_tensor = torch.tensor(indices).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        preds = torch.argmax(outputs, dim=2)[0][:len(words)]
    return [(w, idx_to_tag[p.item()]) for w, p in zip(words, preds)]

def plot_training_progress(losses, accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(losses)
    ax1.set_title("Training Loss")
    ax2.plot(accs)
    ax2.set_title("Validation Accuracy")
    plt.show()

if __name__ == "__main__":
    # Set fraction here - e.g. 0.1 = 10% of data for faster training
    model, word_to_idx, tag_to_idx, idx_to_tag, losses, accs = train_ner_model(train_fraction=0.25)
    plot_training_progress(losses, accs)

    test_sentences = [
        "Barack Obama visited Paris",
        "Amazon CEO Jeff Bezos lives in Seattle",
        "Apple announced a new iPhone today",
    ]
    for sent in test_sentences:
        print(f"\nSentence: {sent}")
        for word, tag in predict_ner(model, sent, word_to_idx, idx_to_tag):
            if tag != 'O':
                print(f"  {word}: {tag}")
