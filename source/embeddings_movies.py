import re
import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import pyro
import pyro.nn as pynn


# 1. Data loading and preprocessing
nltk.download("movie_reviews", quiet=True)
from nltk.corpus import movie_reviews


def load_movie_review_tokens(max_documents=400, min_token_length=2):
    """
    Load a subset of the NLTK movie_reviews corpus and return a list of
    tokenized documents.
    """
    fileids = movie_reviews.fileids()[:max_documents]
    documents = []

    for fid in fileids:
        raw_words = movie_reviews.words(fid)
        tokens = []
        for w in raw_words:
            token = re.sub(r"[^a-z]", "", w.lower())
            if len(token) >= min_token_length:
                tokens.append(token)
        if tokens:
            documents.append(tokens)

    return documents


def build_vocabulary(tokenized_documents, min_count=5, max_vocab_size=2000):
    """
    Build a vocabulary from tokenized documents with a minimum count threshold.
    Reserve index 0 for <UNK>.
    """
    counter = Counter()
    for doc in tokenized_documents:
        counter.update(doc)

    most_common = [
        (token, freq)
        for token, freq in counter.most_common(max_vocab_size)
        if freq >= min_count
    ]

    vocab = ["<UNK>"] + [token for token, _ in most_common]
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for word, i in word_to_idx.items()}

    return vocab, word_to_idx, idx_to_word, counter


def numericalize_documents(tokenized_documents, word_to_idx):
    """
    Convert each tokenized document into a sequence of integer indices.
    """
    unk = word_to_idx["<UNK>"]
    numeric_docs = []
    for doc in tokenized_documents:
        numeric_docs.append([word_to_idx.get(token, unk) for token in doc])
    return numeric_docs


def generate_skipgram_pairs(numeric_documents, window_size=2):
    """
    Generate skip-gram (center, context) pairs from indexed documents.
    """
    pairs = []
    for doc in numeric_documents:
        n = len(doc)
        for t, center in enumerate(doc):
            left = max(0, t - window_size)
            right = min(n, t + window_size + 1)
            for j in range(left, right):
                if j != t:
                    context = doc[j]
                    pairs.append((center, context))
    return pairs


# 2. Pyro model (deterministic: no priors)
class SkipGramPyro(pynn.PyroModule):
    """
    Deterministic skip-gram model in Pyro/PyTorch.
    No priors are assigned; parameters are learned by gradient descent.
    """

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = pynn.PyroModule[nn.Embedding](vocab_size, embedding_dim)
        self.output = pynn.PyroModule[nn.Linear](embedding_dim, vocab_size, bias=False)

        # Small random initialization
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.05)

    def forward(self, center_words):
        """
        center_words: tensor of shape (batch_size,)
        returns logits of shape (batch_size, vocab_size)
        """
        h = self.embedding(center_words)
        logits = self.output(h)
        return logits


# 3. Mini-batching utilities
def batch_iterator(pairs, batch_size=256, shuffle=True):
    """
    Yield mini-batches of (center, context) pairs.
    """
    if shuffle:
        random.shuffle(pairs)

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        centers = torch.tensor([x for x, _ in batch], dtype=torch.long)
        contexts = torch.tensor([y for _, y in batch], dtype=torch.long)
        yield centers, contexts

# 4. Training
def train_skipgram_model(
    pairs,
    vocab_size,
    embedding_dim=2,
    epochs=10,
    batch_size=256,
    learning_rate=0.01,
    device=None,
):
    """
    Train the skip-gram model using cross-entropy loss.
    """
    pyro.clear_param_store()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SkipGramPyro(vocab_size=vocab_size, embedding_dim=embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for centers, contexts in batch_iterator(pairs, batch_size=batch_size, shuffle=True):
            centers = centers.to(device)
            contexts = contexts.to(device)

            optimizer.zero_grad()
            logits = model(centers)
            loss = criterion(logits, contexts)
            loss.backward()
            optimizer.step()

            batch_n = centers.size(0)
            total_loss += loss.item() * batch_n
            total_examples += batch_n

        avg_loss = total_loss / max(total_examples, 1)
        history.append(avg_loss)
        print(f"Epoch {epoch:02d} | Average loss: {avg_loss:.4f}")

    return model, history

# 5. Analysis helpers
def cosine_similarity(vec_a, vec_b, eps=1e-12):
    """
    Compute cosine similarity between two vectors.
    """
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)

    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b) + eps
    return float(num / den)


def most_similar_words(model, word, word_to_idx, idx_to_word, top_k=10):
    """
    Return the top_k most similar words to a given word based on
    cosine similarity of learned embeddings.
    """
    if word not in word_to_idx:
        raise ValueError(f"Word '{word}' is not in the vocabulary.")

    emb = model.embedding.weight.detach().cpu().numpy()
    query_idx = word_to_idx[word]
    query_vec = emb[query_idx]

    similarities = []
    for idx in range(len(idx_to_word)):
        if idx == query_idx:
            continue
        sim = cosine_similarity(query_vec, emb[idx])
        similarities.append((idx_to_word[idx], sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def plot_embeddings_2d(model, idx_to_word, selected_words=None, max_words=80):
    """
    Plot 2D embeddings. If selected_words is None, plot the first max_words
    non-UNK words.
    """
    emb = model.embedding.weight.detach().cpu().numpy()

    if emb.shape[1] != 2:
        raise ValueError("This plot function requires embedding_dim=2.")

    if selected_words is None:
        selected_indices = list(range(1, min(max_words + 1, len(idx_to_word))))
    else:
        inverse_map = {v: k for k, v in idx_to_word.items()}
        selected_indices = [inverse_map[w] for w in selected_words if w in inverse_map]

    plt.figure(figsize=(10, 8))
    for idx in selected_indices:
        x, y = emb[idx]
        word = idx_to_word[idx]
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, word, fontsize=9)

    plt.xlabel("Embedding dimension 1")
    plt.ylabel("Embedding dimension 2")
    plt.title("Learned 2D word embeddings")
    plt.grid(True)
    plt.show()

# 6. End-to-end example
documents = load_movie_review_tokens(max_documents=400, min_token_length=2)
vocab, word_to_idx, idx_to_word, token_counts = build_vocabulary(
    documents,
    min_count=5,
    max_vocab_size=2000,
)

numeric_documents = numericalize_documents(documents, word_to_idx)
pairs = generate_skipgram_pairs(numeric_documents, window_size=2)

print(f"Number of documents: {len(documents)}")
print(f"Vocabulary size: {len(vocab)}")
print(f"Number of skip-gram pairs: {len(pairs)}")

model, loss_history = train_skipgram_model(
    pairs=pairs,
    vocab_size=len(vocab),
    embedding_dim=2,
    epochs=12,
    batch_size=512,
    learning_rate=0.01,
)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Average cross-entropy loss")
plt.title("Training loss")
plt.grid(True)
plt.show()

query_words = ["film", "movie", "story", "character", "good", "bad"]

for query in query_words:
    if query in word_to_idx:
        neighbors = most_similar_words(model, query, word_to_idx, idx_to_word, top_k=8)
        print(f"\nMost similar words to '{query}':")
        for neighbor, sim in neighbors:
            print(f"  {neighbor:15s}  cosine={sim:.4f}")

plot_words = [
    "film", "movie", "story", "character", "plot",
    "good", "bad", "great", "funny", "boring",
    "love", "man", "woman", "family", "comedy",
    "drama", "action", "performance", "director", "music"
]

plot_embeddings_2d(model, idx_to_word, selected_words=plot_words)