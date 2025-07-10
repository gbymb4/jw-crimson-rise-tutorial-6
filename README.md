# Deep Learning with PyTorch - Session 6: Word Embeddings and Named Entity Recognition

## Session Timeline

| Time      | Activity                                    |
| --------- | ------------------------------------------- |
| 0:00 - 0:10 | 1. Check-in + Session 5 Recap              |
| 0:10 - 0:25 | 2. Introduction to Word Embeddings         |
| 0:25 - 0:50 | 3. PyTorch Embedding Layer & NER Dataset   |
| 0:50 - 1:15 | 4. Building and Training NER Model         |
| 1:15 - 1:30 | 5. Results Analysis & Next Steps           |

---

## 1. Check-in + Session 5 Recap

### Quick Recap Questions

* How did your TF-IDF classifier perform compared to the CNN? What surprised you?
* What specific limitations did you notice with bag-of-words representations?
* Can you explain why CNNs struggled with text compared to images?
* What preprocessing steps had the biggest impact on performance?

### Key Takeaways from Session 5

* **Text Representation Challenges**: Sparse vectors, no semantic understanding
* **Architecture Mismatches**: MLPs and CNNs not designed for language structure
* **Fundamental Limitations**: No handling of synonyms, word order, or context
* **Performance Patterns**: TF-IDF often outperformed more complex architectures

---

## 2. Introduction to Word Embeddings

### The Problem with One-Hot Encoding

Traditional text representations treat words as discrete, unrelated symbols:
- `"king"` → `[0, 0, 1, 0, 0, ...]`
- `"queen"` → `[0, 1, 0, 0, 0, ...]`
- No relationship captured between semantically similar words

### Word Embeddings: Dense Semantic Representations

**Key Idea**: Map words to dense, low-dimensional vectors where semantic similarity is preserved.

**Example Relationships**:
- `king - man + woman ≈ queen`
- `paris - france + italy ≈ rome`
- Words with similar meanings have similar vectors

### PyTorch Embedding Layer

**What it does**: Converts integer indices to dense vectors through a learned lookup table.

**Mathematical Operation**:
```
Input: word_index (integer)
Output: embedding_vector (dense float vector)
```

**Key Parameters**:
- `num_embeddings`: Size of vocabulary (how many unique words)
- `embedding_dim`: Size of each embedding vector (typically 50-300)
- `padding_idx`: Index for padding tokens (usually 0)

---

## 3. PyTorch Embedding Layer & NER Dataset

### Named Entity Recognition (NER)

**Task**: Identify and classify named entities in text (person, location, organization, etc.)

**Example**:
- Input: `"Barack Obama visited Paris last week"`
- Output: `"[Barack Obama](PERSON) visited [Paris](LOCATION) last week"`

### Why NER for Learning Embeddings?

1. **Word-level task**: Each word gets its own prediction
2. **Context matters**: Same word can be different entity types
3. **Manageable complexity**: Simpler than full sentence understanding
4. **Clear evaluation**: Easy to measure token-level accuracy

---

## 4. Building and Training NER Model

### Complete Implementation
```import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import requests
import io

# Download and prepare CoNLL-2003 NER dataset
def download_conll_data():
    """Download CoNLL-2003 NER dataset from a public source"""
    # Using a simplified version of CoNLL-2003 data
    # In practice, you'd download from: https://www.clips.uantwerpen.be/conll2003/ner/
    
    # Sample data for demonstration (replace with actual download)
    sample_data = [
        ("EU", "B-ORG"),
        ("rejects", "O"),
        ("German", "B-MISC"),
        ("call", "O"),
        ("to", "O"),
        ("boycott", "O"),
        ("British", "B-MISC"),
        ("lamb", "O"),
        (".", "O"),
        ("Peter", "B-PER"),
        ("Blackburn", "I-PER"),
        ("BRUSSELS", "B-LOC"),
        ("1996-08-22", "O"),
        ("The", "O"),
        ("European", "B-ORG"),
        ("Commission", "I-ORG"),
        ("said", "O"),
        ("on", "O"),
        ("Thursday", "O"),
        ("it", "O"),
        ("disagreed", "O"),
        ("with", "O"),
        ("German", "B-MISC"),
        ("advice", "O"),
        ("to", "O"),
        ("consumers", "O"),
        ("to", "O"),
        ("shun", "O"),
        ("British", "B-MISC"),
        ("lamb", "O"),
        ("until", "O"),
        ("scientists", "O"),
        ("determine", "O"),
        ("whether", "O"),
        ("mad", "O"),
        ("cow", "O"),
        ("disease", "O"),
        ("can", "O"),
        ("be", "O"),
        ("transmitted", "O"),
        ("to", "O"),
        ("sheep", "O"),
        (".", "O"),
    ]
    
    # Create sentences by splitting on periods
    sentences = []
    current_sentence = []
    
    for word, tag in sample_data:
        if word == "." and current_sentence:
            current_sentence.append((word, tag))
            sentences.append(current_sentence)
            current_sentence = []
        else:
            current_sentence.append((word, tag))
    
    # Add more sample sentences for better training
    additional_sentences = [
        [("John", "B-PER"), ("Smith", "I-PER"), ("works", "O"), ("at", "O"), ("Google", "B-ORG"), ("in", "O"), ("California", "B-LOC"), (".", "O")],
        [("Apple", "B-ORG"), ("Inc", "I-ORG"), ("is", "O"), ("located", "O"), ("in", "O"), ("Cupertino", "B-LOC"), (".", "O")],
        [("Microsoft", "B-ORG"), ("CEO", "O"), ("Satya", "B-PER"), ("Nadella", "I-PER"), ("announced", "O"), ("new", "O"), ("products", "O"), (".", "O")],
        [("The", "O"), ("New", "B-LOC"), ("York", "I-LOC"), ("Times", "I-ORG"), ("reported", "O"), ("yesterday", "O"), (".", "O")],
        [("Amazon", "B-ORG"), ("founder", "O"), ("Jeff", "B-PER"), ("Bezos", "I-PER"), ("visited", "O"), ("Seattle", "B-LOC"), (".", "O")],
    ]
    
    sentences.extend(additional_sentences)
    return sentences

# Dataset class
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
        
        # Convert words and tags to indices
        word_indices = []
        tag_indices = []
        
        for word, tag in sentence:
            word_idx = self.word_to_idx.get(word.lower(), self.word_to_idx['<UNK>'])
            tag_idx = self.tag_to_idx[tag]
            word_indices.append(word_idx)
            tag_indices.append(tag_idx)
        
        # Pad sequences
        while len(word_indices) < self.max_length:
            word_indices.append(self.word_to_idx['<PAD>'])
            tag_indices.append(self.tag_to_idx['O'])  # Pad with 'O' (outside) tag
        
        # Truncate if too long
        word_indices = word_indices[:self.max_length]
        tag_indices = tag_indices[:self.max_length]
        
        return torch.tensor(word_indices), torch.tensor(tag_indices)

# NER Model with Embedding Layer
class NERModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, dropout=0.3):
        super(NERModel, self).__init__()
        
        # Embedding layer: converts word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Simple feedforward network for each word
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, num_tags)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embeddings = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        
        # Apply feedforward network to each word independently
        hidden = self.relu(self.hidden(embeddings))  # (batch_size, sequence_length, hidden_dim)
        hidden = self.dropout(hidden)
        output = self.output(hidden)  # (batch_size, sequence_length, num_tags)
        
        return output

# Training function
def train_ner_model():
    print("Loading NER dataset...")
    sentences = download_conll_data()
    
    # Build vocabularies
    word_counter = Counter()
    tag_counter = Counter()
    
    for sentence in sentences:
        for word, tag in sentence:
            word_counter[word.lower()] += 1
            tag_counter[tag] += 1
    
    # Create word-to-index mappings
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counter.most_common():
        word_to_idx[word] = len(word_to_idx)
    
    tag_to_idx = {}
    for tag, count in tag_counter.most_common():
        tag_to_idx[tag] = len(tag_to_idx)
    
    idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
    
    print(f"Vocabulary size: {len(word_to_idx)}")
    print(f"Number of NER tags: {len(tag_to_idx)}")
    print(f"Tags: {list(tag_to_idx.keys())}")
    
    # Split data (simple split for demo)
    train_size = int(0.8 * len(sentences))
    train_sentences = sentences[:train_size]
    val_sentences = sentences[train_size:]
    
    # Create datasets
    train_dataset = NERDataset(train_sentences, word_to_idx, tag_to_idx)
    val_dataset = NERDataset(val_sentences, word_to_idx, tag_to_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Initialize model
    model = NERModel(
        vocab_size=len(word_to_idx),
        embedding_dim=100,
        hidden_dim=128,
        num_tags=len(tag_to_idx)
    )
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=tag_to_idx['O'])  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 50
    train_losses = []
    val_accuracies = []
    
    print("\nTraining model...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        for batch_words, batch_tags in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_words)  # (batch_size, seq_len, num_tags)
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, len(tag_to_idx))  # (batch_size * seq_len, num_tags)
            batch_tags = batch_tags.view(-1)  # (batch_size * seq_len)
            
            loss = criterion(outputs, batch_tags)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_words, batch_tags in val_loader:
                outputs = model(batch_words)
                predictions = torch.argmax(outputs, dim=2)
                
                # Only count non-padding tokens
                mask = batch_tags != tag_to_idx['O']
                correct += ((predictions == batch_tags) & mask).sum().item()
                total += mask.sum().item()
        
        accuracy = correct / total if total > 0 else 0
        train_losses.append(total_loss / len(train_loader))
        val_accuracies.append(accuracy)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:2d}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {accuracy:.4f}')
    
    return model, word_to_idx, tag_to_idx, idx_to_tag, train_losses, val_accuracies

# Prediction function
def predict_ner(model, sentence, word_to_idx, idx_to_tag, max_length=50):
    """Predict NER tags for a given sentence"""
    model.eval()
    
    # Tokenize and convert to indices
    words = sentence.lower().split()
    word_indices = []
    
    for word in words:
        word_idx = word_to_idx.get(word, word_to_idx['<UNK>'])
        word_indices.append(word_idx)
    
    # Pad sequence
    while len(word_indices) < max_length:
        word_indices.append(word_to_idx['<PAD>'])
    
    # Convert to tensor
    input_tensor = torch.tensor(word_indices).unsqueeze(0)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        predictions = torch.argmax(outputs, dim=2)
    
    # Convert back to tags
    predicted_tags = []
    for i in range(len(words)):
        tag_idx = predictions[0][i].item()
        predicted_tags.append(idx_to_tag[tag_idx])
    
    return list(zip(words, predicted_tags))

# Visualization function
def plot_training_progress(train_losses, val_accuracies):
    """Plot training loss and validation accuracy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Embedding visualization
def visualize_embeddings(model, word_to_idx, words_to_show=20):
    """Visualize word embeddings using PCA"""
    from sklearn.decomposition import PCA
    
    # Get embedding weights
    embeddings = model.embedding.weight.data.numpy()
    
    # Select interesting words to visualize
    interesting_words = ['google', 'apple', 'microsoft', 'amazon', 'john', 'peter', 'california', 'seattle', 'new', 'york']
    word_indices = []
    word_labels = []
    
    for word in interesting_words:
        if word in word_to_idx:
            word_indices.append(word_to_idx[word])
            word_labels.append(word)
    
    if len(word_indices) < 2:
        print("Not enough words in vocabulary for visualization")
        return
    
    # Apply PCA
    selected_embeddings = embeddings[word_indices]
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(selected_embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    
    for i, word in enumerate(word_labels):
        plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
    
    plt.title('Word Embeddings Visualization (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Train the model
    model, word_to_idx, tag_to_idx, idx_to_tag, losses, accuracies = train_ner_model()
    
    # Plot training progress
    plot_training_progress(losses, accuracies)
    
    # Test predictions
    print("\n" + "="*50)
    print("TESTING NER PREDICTIONS")
    print("="*50)
    
    test_sentences = [
        "Barack Obama visited Paris last week",
        "Google CEO works in California",
        "Microsoft announced new products yesterday",
        "John Smith lives in New York"
    ]
    
    for sentence in test_sentences:
        predictions = predict_ner(model, sentence, word_to_idx, idx_to_tag)
        print(f"\nSentence: {sentence}")
        print("Predictions:")
        for word, tag in predictions:
            if tag != 'O':  # Only show non-'O' tags
                print(f"  {word}: {tag}")
    
    # Visualize embeddings
    print("\n" + "="*50)
    print("WORD EMBEDDINGS VISUALIZATION")
    print("="*50)
    try:
        visualize_embeddings(model, word_to_idx)
    except ImportError:
        print("sklearn not available for embedding visualization")
    
    # Show embedding layer details
    print("\n" + "="*50)
    print("EMBEDDING LAYER ANALYSIS")
    print("="*50)
    
    print(f"Embedding layer shape: {model.embedding.weight.shape}")
    print(f"Each word is represented by a {model.embedding.weight.shape[1]}-dimensional vector")
    
    # Show similarity between some words
    def cosine_similarity(vec1, vec2):
        return torch.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
    
    embeddings = model.embedding.weight.data
    if 'google' in word_to_idx and 'apple' in word_to_idx:
        google_emb = embeddings[word_to_idx['google']]
        apple_emb = embeddings[word_to_idx['apple']]
        similarity = cosine_similarity(google_emb, apple_emb)
        print(f"\nSimilarity between 'google' and 'apple': {similarity:.3f}")
    
    if 'john' in word_to_idx and 'peter' in word_to_idx:
        john_emb = embeddings[word_to_idx['john']]
        peter_emb = embeddings[word_to_idx['peter']]
        similarity = cosine_similarity(john_emb, peter_emb)
        print(f"Similarity between 'john' and 'peter': {similarity:.3f}")
    
    print("\nTraining complete! The model has learned to:")
    print("1. Convert words to dense embedding vectors")
    print("2. Use these embeddings to predict named entity tags")
    print("3. Capture semantic relationships between words")
```

## 5. Results Analysis & Next Steps

### What We Accomplished

**Embedding Layer Understanding:**
- Words converted from sparse one-hot vectors to dense 100-dimensional embeddings
- Each word learns a unique representation that captures semantic properties
- Similar words (like company names) develop similar embedding vectors

**NER Task Performance:**
- Simple feedforward network processes each word independently
- Embeddings provide richer input than one-hot encoding
- Model learns to distinguish between different entity types

### Key Observations

**Expected Results:**
- Training accuracy: ~70-85% (depends on dataset size)
- Common entities (PERSON, ORGANIZATION) often classified correctly
- Rare entities may be missed due to limited training data

**Embedding Insights:**
- Company names (Google, Apple, Microsoft) cluster together in embedding space
- Person names (John, Peter) show similarity
- Location names develop distinct patterns

### Limitations of This Approach

**Why This Model is Still Limited:**
1. **No Context**: Each word classified independently
2. **No Sequence Modeling**: Cannot use surrounding words effectively
3. **Limited Training Data**: Small dataset limits embedding quality
4. **Architecture Mismatch**: Feedforward network not ideal for sequences

### Next Steps: Beyond Basic Embeddings

**Immediate Improvements:**
- **Pretrained Embeddings**: Use Word2Vec, GloVe embeddings trained on large corpora
- **Context Windows**: Include surrounding words as features
- **Sequence Models**: RNNs, LSTMs, or Transformers for better sequence understanding

**Advanced Techniques:**
- **Bidirectional Models**: Process sequences in both directions
- **Attention Mechanisms**: Focus on relevant parts of input
- **Contextual Embeddings**: ELMo, BERT-style representations

### Discussion Questions

1. **Embedding Quality**: How might we improve the quality of our word embeddings with limited data?

2. **Architecture Choice**: Why might a simple feedforward network struggle with NER compared to sequence models?

3. **Evaluation**: What metrics beyond accuracy would be useful for NER evaluation?

4. **Pretrained vs. Custom**: When would you use pretrained embeddings vs. training your own?

### Homework

**Basic Tasks:**
- Experiment with different embedding dimensions (50, 200, 300)
- Add more training data and observe embedding changes
- Try the model on different types of text

**Intermediate Tasks:**
- Implement a context window approach (use surrounding words)
- Download and use pretrained Word2Vec embeddings
- Visualize how embeddings change during training

**Advanced Tasks:**
- Implement a simple RNN for sequence modeling
- Compare performance with different sequence lengths
- Analyze what linguistic patterns the embeddings capture

---

This session provides a practical introduction to word embeddings through a concrete NER task, demonstrating both the power and limitations of embedding representations while setting up the foundation for more advanced sequence modeling in future sessions.