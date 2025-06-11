import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import wandb
from tokenizer import Tokenizer, Cirilica


CONFIG = {
        'tokenizer': Cirilica,
        'trained_tokenizer': None,
        'tokenizer_save': "model/cirilica_tokenizer",
        'vocab_size': 500,
        'batch_size': 64,
        'block_size': 256,
        'max_iters': 2,
        'eval_interval': 1,
        'learning_rate': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'eval_iters': 200,
        'n_embd': 384,
        'n_heads': 6,
        'n_layers': 6,
        'dropout': 0.2,
        'generate_length': 2000,
        'input_file': 'data/test_text.txt',
        'output_file': 'output/generated_text_cirilica_tokenizer.txt',
        'model_save_path': 'model/model_checkpoint_cirilica.pt'
    }


def encode(s, stoi):
    return [stoi[c] for c in s]

def decode(l, itos):
    return ''.join([itos[i] for i in l])



class AttentionHead(nn.Module):
    """ one head of self-attention """
    def __init__(self, n_embd, block_size, head_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention weights
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, n_embd, block_size, num_heads, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(n_embd, block_size, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = self.net(x)
        return x


class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, block_size, n_heads, dropout):
        super().__init__()
        head_size = n_embd // n_heads
        self.multi_head_attention = MultiHeadAttention(n_embd, block_size, n_heads, head_size, dropout)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.multi_head_attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


# GPT model
class GPT(nn.Module):

    def __init__(self, 
                 tokenizer: Tokenizer, 
                 block_size = CONFIG['block_size'], 
                 n_embd = CONFIG['n_embd'], 
                 n_layers = CONFIG['n_layers'], 
                 n_heads = CONFIG['n_heads'], 
                 dropout = CONFIG['dropout']):
        super().__init__()
        vocab_size = tokenizer.vocab_size
        self.tokenizer = tokenizer
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(n_embd, block_size, n_heads, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_embd = self.token_embedding_table(idx) # (B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_embd + pos_embd # (B,T,C)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, block_size, max_new_tokens):

        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

def train_tokenizer(text):
    tokenizer = CONFIG['tokenizer']()
    tokenizer.train(text, CONFIG['vocab_size'])
    return tokenizer    

def load_data(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Load tokenizer
    if not CONFIG['trained_tokenizer']:
        tokenizer = train_tokenizer(text)
    else:
        tokenizer:Tokenizer = CONFIG['tokenizer']()
        tokenizer.load(CONFIG['trained_tokenizer'])
     
    # Train and test splits

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, tokenizer

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, eval_iters, train_data, val_data, block_size, batch_size, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        data = train_data if split == 'train' else val_data
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()     
        out[split] = losses.mean()
    model.train()
    return out



def train_model(model, train_data, val_data, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Initialize wandb
    wandb.init(
        project="EpskiGPT",
        config=config
    )

    time_start = time.time()

    for iter in range(config['max_iters']):
        if iter % config['eval_interval'] == 0:
            losses = estimate_loss(
                model, config['eval_iters'], 
                train_data, val_data,
                config['block_size'], config['batch_size'],
                config['device']
            )
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            wandb.log({
                "train_loss": losses['train'],
                "val_loss": losses['val'],
                "iteration": iter,
            })

        xb, yb = get_batch(train_data, config['block_size'], 
                          config['batch_size'], config['device'])
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        wandb.log({
            "batch_loss": loss.item(),
            "iteration": iter,
        })

    time_end = time.time()
    time_needed = time_end - time_start
    print(f"Time taken: {time.strftime('%H:%M:%S', time.gmtime(time_needed))}")
    
    wandb.finish()
    return model, optimizer, loss

def save_model(model, optimizer, loss, iter, config):
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter': iter,
        'loss': loss,
    }, config['model_save_path'])
    print(f"Model saved to {config['model_save_path']}")



def generate_output(model: GPT, config: dict, tokenizer: Tokenizer):
    # Generate text
    context = torch.zeros((1, 1), dtype=torch.long, device=config['device'])
    generated_text = tokenizer.decode(model.generate(context, block_size=config['block_size'],
                          max_new_tokens=config['generate_length'])[0].tolist())

    # Save generated text
    output_file = config['output_file']
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(generated_text)
    print(f"Generated text saved to {output_file}")


def main():
    # Configuration
    config = CONFIG

    # Set random seed
    torch.manual_seed(1337)

    # Load and prepare data
    train_data, val_data, tokenizer = load_data(
        config['input_file']
    )

    # Initialize model
    model = GPT(
        tokenizer=tokenizer,
        block_size=config['block_size'],
        n_embd=config['n_embd'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        dropout=config['dropout']
    )

    model.to(config['device'])

    # Train model
    model, optimizer, loss = train_model(model, train_data, val_data, config)

    # Save model and generate text
    save_model(model, optimizer, loss, config['max_iters'], config)
    generate_output(model, config, tokenizer)
    tokenizer.save(config['tokenizer_save'])

if __name__ == '__main__':
    main()