# -*- coding: utf-8 -*-
"""
NER Homework: From Feedforward to Sequence Models (LSTM/GRU)

ASSIGNMENT: Improve NER performance by implementing sequence-aware models

Your task is to replace the simple feedforward architecture with LSTM/GRU models
that can better capture sequential dependencies in text data.

Expected improvements:
- Better context understanding
- Improved handling of multi-token entities  
- Higher accuracy on validation set
- Better generalization to unseen text

Complete all TODO sections to implement and compare different architectures.
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

# =============================================================================
# BASELINE MODEL (Already implemented - DO NOT MODIFY)
# =============================================================================
class BaselineNERModel(nn.Module):
    """
    Baseline feedforward model - processes each word independently
    This is the model from your session that you need to improve upon
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        super(BaselineNERModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(hidden_dim, num_tags)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = self.relu(self.hidden(x))  # Each word processed independently
        x = self.dropout(x)
        return self.output(x)  # (batch_size, seq_len, num_tags)

# =============================================================================
# TODO 1: Implement LSTM-based NER Model
# =============================================================================
class LSTMNERModel(nn.Module):
    """
    LSTM-based NER model that can capture sequential dependencies
    
    This model should improve upon the baseline by:
    - Processing sequences rather than individual words
    - Capturing context from surrounding words
    - Better handling of entity boundaries
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, num_layers=2, dropout=0.3):
        super(LSTMNERModel, self).__init__()
        
        # TODO 1a: Set up the embedding layer
        # Consider: What parameters does nn.Embedding need?
        # YOUR CODE HERE
        pass
        
        # TODO 1b: Set up the LSTM layer
        # Consider: What makes an LSTM good for sequence processing?
        # Think about: input size, hidden size, directionality, layers
        # YOUR CODE HERE
        pass
        
        # TODO 1c: Set up the output projection layer
        # Consider: What is the output size of your LSTM?
        # YOUR CODE HERE
        pass
    
    def forward(self, x):
        # TODO 1d: Implement the forward pass
        # Think about: How does data flow through embeddings -> LSTM -> output?
        # YOUR CODE HERE
        pass

# =============================================================================
# TODO 2: Implement GRU-based NER Model
# =============================================================================
class GRUNERModel(nn.Module):
    """
    GRU-based NER model as an alternative to LSTM
    
    GRU is similar to LSTM but with some differences:
    - Fewer parameters (potentially faster)
    - Different gating mechanism
    - Often performs similarly to LSTM
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, num_layers=2, dropout=0.3):
        super(GRUNERModel, self).__init__()
        
        # TODO 2a: Set up the embedding layer
        # YOUR CODE HERE
        pass
        
        # TODO 2b: Set up the GRU layer
        # Consider: How does GRU initialization differ from LSTM?
        # YOUR CODE HERE
        pass
        
        # TODO 2c: Set up the output projection layer
        # YOUR CODE HERE
        pass
    
    def forward(self, x):
        # TODO 2d: Implement the forward pass
        # YOUR CODE HERE
        pass

# =============================================================================
# TODO 3: Implement Training Function with Model Comparison
# =============================================================================
def train_and_compare_models(train_fraction=0.25):
    """
    Train multiple models and compare their performance
    
    Returns:
        Dictionary with model results for comparison
    """
    
    # Load and prepare data (same as before)
    sentences = download_conll_data()
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

    # TODO 3a: Initialize all models for comparison
    models = {
        'Baseline': BaselineNERModel(len(word_to_idx), 100, 128, len(tag_to_idx)),
        # Add your implemented models here
        # Consider: What parameters should you use for fair comparison?
    }
    
    results = {}
    
    # TODO 3b: Training loop for each model
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name} Model")
        print('='*50)
        
        model = model.to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=tag_to_idx['O'])
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_losses, val_accuracies = [], []
        best_val_acc = 0.0
        
        # TODO 3c: Implement training loop
        for epoch in range(30):
            model.train()
            total_loss = 0
            
            for batch_words, batch_tags in train_loader:
                batch_words, batch_tags = batch_words.to(device), batch_tags.to(device)
                
                # TODO: Implement training step
                # Consider: forward pass, loss calculation, backpropagation
                # YOUR CODE HERE
                pass
            
            # TODO 3d: Implement validation loop
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch_words, batch_tags in val_loader:
                    batch_words, batch_tags = batch_words.to(device), batch_tags.to(device)
                    
                    # TODO: Implement validation step
                    # Consider: How do you calculate accuracy for sequence labeling?
                    # YOUR CODE HERE
                    pass
            
            acc = correct / total if total > 0 else 0
            train_losses.append(total_loss / len(train_loader))
            val_accuracies.append(acc)
            
            if acc > best_val_acc:
                best_val_acc = acc
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:02d}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {acc:.4f}")
        
        results[model_name] = {
            'model': model,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc
        }
        
        print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return results, word_to_idx, tag_to_idx, idx_to_tag

# =============================================================================
# TODO 4: Implement Performance Analysis
# =============================================================================
def analyze_model_performance(results):
    """
    Analyze and visualize model performance comparison
    """
    
    # Create training curves visualization
    num_models = len(results)
    fig, axes = plt.subplots(2, num_models, figsize=(5*num_models, 8))
    
    if num_models == 1:
        axes = axes.reshape(2, 1)
    
    for i, (model_name, result) in enumerate(results.items()):
        # Plot training loss
        axes[0, i].plot(result['train_losses'])
        axes[0, i].set_title(f'{model_name} - Training Loss')
        axes[0, i].set_xlabel('Epoch')
        axes[0, i].set_ylabel('Loss')
        axes[0, i].grid(True)
        
        # Plot validation accuracy
        axes[1, i].plot(result['val_accuracies'])
        axes[1, i].set_title(f'{model_name} - Validation Accuracy')
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].set_ylabel('Accuracy')
        axes[1, i].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Create performance comparison bar chart
    model_names = list(results.keys())
    best_accuracies = [results[name]['best_val_acc'] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, best_accuracies, color=['red', 'blue', 'green', 'orange'][:len(model_names)])
    plt.title('Model Performance Comparison')
    plt.ylabel('Best Validation Accuracy')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, best_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    # Print performance analysis
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    best_model = max(results.items(), key=lambda x: x[1]['best_val_acc'])
    print(f"Best performing model: {best_model[0]} (Accuracy: {best_model[1]['best_val_acc']:.4f})")
    
    # Compare with baseline if available
    if 'Baseline' in results:
        baseline_acc = results['Baseline']['best_val_acc']
        print(f"Baseline accuracy: {baseline_acc:.4f}")
        print("\nImprovements over baseline:")
        for name, result in results.items():
            if name != 'Baseline':
                improvement = result['best_val_acc'] - baseline_acc
                print(f"  {name}: +{improvement:.4f} ({improvement/baseline_acc*100:.1f}% relative)")
    
    # Analyze training stability
    print("\nTraining Stability Analysis:")
    for name, result in results.items():
        final_losses = result['train_losses'][-5:]  # Last 5 epochs
        loss_std = np.std(final_losses)
        print(f"  {name}: Final loss std = {loss_std:.4f} ({'stable' if loss_std < 0.1 else 'unstable'})")
    
    # Analyze convergence speed
    print("\nConvergence Analysis:")
    for name, result in results.items():
        val_accs = result['val_accuracies']
        best_epoch = np.argmax(val_accs)
        print(f"  {name}: Best accuracy reached at epoch {best_epoch}")
        
        # Check if model improved significantly after epoch 10
        if len(val_accs) > 10:
            early_acc = max(val_accs[:10])
            late_improvement = result['best_val_acc'] - early_acc
            print(f"    Improvement after epoch 10: +{late_improvement:.4f}")
    
    print("\nKey Observations:")
    print("- Sequence models (LSTM/GRU) should outperform the baseline")
    print("- Look for smoother training curves in sequence models")
    print("- Better models often show more consistent validation improvements")

# =============================================================================
# Implement Sequence-Aware Prediction Analysis
# =============================================================================
def analyze_predictions(results, word_to_idx, tag_to_idx, idx_to_tag):
    """
    Analyze how different models handle sequence prediction
    """
    
    # Test sentences that require sequence understanding
    test_sentences = [
        "John Smith works at Apple Inc in California",
        "New York Times reported that Microsoft CEO visited Google",
        "Barack Obama and Hillary Clinton met in Washington",
        "Amazon founder Jeff Bezos lives in Seattle Washington",
    ]
    
    print("\n" + "="*60)
    print("SEQUENCE PREDICTION ANALYSIS")
    print("="*60)
    
    # Compare model predictions on test sentences
    for sentence in test_sentences:
        print(f"\nSentence: {sentence}")
        print("-" * 50)
        
        for model_name, result in results.items():
            model = result['model']
            predictions = predict_ner(model, sentence, word_to_idx, idx_to_tag)
            
            print(f"\n{model_name} predictions:")
            entities = [(word, tag) for word, tag in predictions if tag != 'O']
            if entities:
                for word, tag in entities:
                    print(f"  {word}: {tag}")
            else:
                print("  No entities found")
    
    # Analyze sequence consistency
    print("\n" + "="*60)
    print("SEQUENCE PATTERN ANALYSIS")
    print("="*60)
    
    # Test multi-token entity handling
    multi_token_test = "Apple Inc CEO Tim Cook announced new products"
    print(f"Multi-token entity test: {multi_token_test}")
    
    for model_name, result in results.items():
        model = result['model']
        predictions = predict_ner(model, multi_token_test, word_to_idx, idx_to_tag)
        
        print(f"\n{model_name}:")
        entities_found = False
        for word, tag in predictions:
            if tag != 'O':
                print(f"  {word}: {tag}")
                entities_found = True
        
        if not entities_found:
            print("  No entities found")
        
        # Check for proper B-I sequences
        tags = [tag for _, tag in predictions]
        b_i_consistency = check_bio_consistency(tags)
        print(f"  B-I tag consistency: {'✓' if b_i_consistency else '✗'}")
    
    # Additional analysis: Multi-token entity detection
    print("\n" + "="*60)
    print("MULTI-TOKEN ENTITY ANALYSIS")
    print("="*60)
    
    multi_token_examples = [
        "Microsoft Corporation",
        "New York",
        "Barack Obama",
        "Google Inc"
    ]
    
    for example in multi_token_examples:
        print(f"\nTesting: {example}")
        for model_name, result in results.items():
            model = result['model']
            predictions = predict_ner(model, example, word_to_idx, idx_to_tag)
            
            tags = [tag for _, tag in predictions]
            multi_token_detected = any(tag.startswith('B-') for tag in tags) and any(tag.startswith('I-') for tag in tags)
            
            print(f"  {model_name}: Multi-token entity detected: {'✓' if multi_token_detected else '✗'}")
    
    print("\nKey Insights:")
    print("- Sequence models should better handle multi-token entities")
    print("- Look for consistent B-I-O tag patterns")
    print("- LSTM/GRU models should show better entity boundary detection")

def check_bio_consistency(tags):
    """
    Check if B-I-O tags follow proper sequence rules
    Rules: I-TAG must follow B-TAG or I-TAG of same entity type
    """
    for i in range(len(tags)):
        if tags[i].startswith('I-'):
            entity_type = tags[i][2:]  # Get entity type (e.g., 'PER' from 'I-PER')
            
            # Check if there's a valid predecessor
            if i == 0:  # I-tag at the beginning is invalid
                return False
            
            prev_tag = tags[i-1]
            if prev_tag == 'O':  # I-tag after O is invalid
                return False
            
            if prev_tag.startswith('B-') or prev_tag.startswith('I-'):
                prev_entity_type = prev_tag[2:]
                if prev_entity_type != entity_type:  # Different entity types
                    return False
    
    return True

def predict_ner(model, sentence, word_to_idx, idx_to_tag):
    """Predict NER tags for a sentence"""
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

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("NER Homework: Sequence Models vs Feedforward")
    print("=" * 60)
    
    # TODO 6: Complete the main execution
    # Consider: What's the right order to run your experiments?
    print("Training and comparing models...")
    
    # YOUR CODE HERE
    pass
    
    print("\nHomework complete! Key questions to consider:")
    print("1. Which model performed best and why?")
    print("2. How do sequence models handle multi-token entities better?")
    print("3. What are the trade-offs between LSTM and GRU?")
    print("4. How does sequence context improve NER performance?")
    print("5. What patterns do you see in the prediction analysis?")

# =============================================================================
# EXPECTED RESULTS AND LEARNING OBJECTIVES
# =============================================================================
"""
EXPECTED PERFORMANCE IMPROVEMENTS:

1. Baseline (Feedforward): ~60-70% accuracy
   - Processes each word independently
   - Struggles with multi-token entities
   - Poor B-I-O tag consistency

2. LSTM Model: ~75-85% accuracy
   - Better context understanding
   - Improved multi-token entity recognition
   - Better sequence consistency

3. GRU Model: ~73-83% accuracy
   - Similar to LSTM but potentially faster
   - Good balance of performance and efficiency

KEY LEARNING OBJECTIVES:
- Understand why sequence models work better for NER
- Learn LSTM/GRU architecture differences
- Practice implementing recurrent neural networks
- Analyze model performance on sequence-specific tasks
- Compare different approaches to sequence modeling

IMPLEMENTATION HINTS:
- Pay attention to tensor shapes throughout your models
- Consider using bidirectional RNNs for better context
- Remember that LSTM/GRU output dimensions depend on bidirectionality
- Think about how padding affects your loss calculation
- Consider the computational trade-offs between different architectures

EVALUATION CRITERIA:
- Correct implementation of LSTM and GRU models
- Proper training loop with loss calculation and backpropagation
- Meaningful performance analysis and visualization
- Insightful comparison of model capabilities
- Clear understanding of sequence modeling benefits
"""