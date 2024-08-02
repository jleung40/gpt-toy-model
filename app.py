from flask import Flask, render_template, request, jsonify
import os  # Make sure to import the os module
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASETS_DIR = 'datasets'
# Define model components
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, _ = self.forward(index_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index
@app.route('/')
def index():
    # Get the list of available datasets
    datasets = [f for f in os.listdir(DATASETS_DIR) if os.path.isfile(os.path.join(DATASETS_DIR, f))]
    return render_template('index.html', datasets=datasets)

@app.route('/train', methods=['POST'])
def train():
    selected_dataset = request.form['dataset']
    dataset_path = os.path.join(DATASETS_DIR, selected_dataset)

    # Set hyperparameters based on user input
    block_size = int(request.form.get('blockSize', block_size))
    max_iters = int(request.form.get('maxIters', max_iters))
    eval_iters = int(request.form.get('evalIters', eval_iters))
    n_embd = int(request.form.get('nEmb', n_embd))
    n_head = int(request.form.get('nHead', n_head))
    n_layer = int(request.form.get('nLayer', n_layer))
    dropout = float(request.form.get('dropout', dropout))
    learning_rate = float(request.form.get('learningRate', learning_rate))

    # Read the dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Process the text into words
    words = text.split()
    vocab_size = len(set(words))
    word_to_int = {word: i for i, word in enumerate(set(words))}
    int_to_word = {i: word for i, word in enumerate(set(words))}
    encode = lambda s: [word_to_int[word] for word in s.split()]
    decode = lambda l: ' '.join([int_to_word[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)

    # Split the text into training and validation sets
    train_text, val_text = train_test_split(text.splitlines(), test_size=0.2, random_state=42)

    # Define functions to get data batches
    def get_random_chunk(split):
        text_data = train_text if split == 'train' else val_text
        random_text = random.choice(text_data)
        data = torch.tensor(encode(random_text), dtype=torch.long)
        return data

    def get_batch(split):
        data = get_random_chunk(split)
        ix = torch.randint(len(data) - block_size, (32,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    # Define model components
    class Head(nn.Module):
        """One head of self-attention"""
        def __init__(self, head_size):
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            B, T, C = x.shape
            k = self.key(x)
            q = self.query(x)
            wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            v = self.value(x)
            out = wei @ v
            return out

    class MultiHeadAttention(nn.Module):
        """Multiple heads of self-attention in parallel"""
        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(head_size * num_heads, n_embd)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out

    class FeedFoward(nn.Module):
        """A simple linear layer followed by a non-linearity"""
        def __init__(self, n_embd):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        """Transformer block: communication followed by computation"""
        def __init__(self, n_embd, n_head):
            super().__init__()
            head_size = n_embd // n_head
            self.sa = MultiHeadAttention(n_head, head_size)
            self.ffwd = FeedFoward(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            y = self.sa(x)
            x = self.ln1(x + y)
            y = self.ffwd(x)
            x = self.ln2(x + y)
            return x

    class GPTLanguageModel(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size)
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, index, targets=None):
            B, T = index.shape
            tok_emb = self.token_embedding_table(index)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
            x = tok_emb + pos_emb
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)

            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                targets = targets.view(B * T)
                loss = F.cross_entropy(logits, targets)

            return logits, loss

        def generate(self, index, max_new_tokens):
            for _ in range(max_new_tokens):
                index_cond = index[:, -block_size:]
                logits, _ = self.forward(index_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                index_next = torch.multinomial(probs, num_samples=1)
                index = torch.cat((index, index_next), dim=1)
            return index

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPTLanguageModel(vocab_size)
    model = model.to(device)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters):
        if iter % eval_iters == 0:
            losses = estimate_loss()
            print(f"step {iter}, train loss {losses['train']:.3f}, val loss {losses['val']:.3f}")

        xb, yb = get_batch('train')
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    def generate_text(starting_text, max_new_tokens=50):
        model.eval()
        starting_index = torch.tensor(encode(starting_text), dtype=torch.long).unsqueeze(0).to(device)
        generated_indices = model.generate(starting_index, max_new_tokens=max_new_tokens)
        generated_text = decode(generated_indices[0].tolist())
        return generated_text

    prompt = "Dorothy began to feel"
    generated_text = generate_text(prompt, max_new_tokens=100)
    print(f"Final generated text:\n{generated_text}")

    return jsonify({"selected_dataset": selected_dataset, "generated_text": generated_text})


if __name__ == '__main__':
    app.run(debug=True)
