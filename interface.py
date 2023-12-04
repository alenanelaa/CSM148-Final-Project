import gradio as gr
import torch
import torch.nn as nn
from torch.nn import functional as F

VALID_END_CHARS = ['.', '!', '?']
MODEL_PATH = './model_128_8_8.pth'

INTERFACE_TITLE = "JokeGPT"
INTERFACE_DESCRIPTION = "Input some starting character(s) or word(s) to create a joke!"
TEXTBOX_PLACEHOLDER = "Input some starting character(s) or word(s)!"
INTERFACE_EXAMPLE_ENTRIES = ["UCLA", "Why did", "What"]

MIN_CHAR_OUTPUT = 175

# hyperparameters
block_size = 128
n_embd = 128
n_head = 8
n_layer = 8
dropout = 0.0

chars = ''
with open('./mini_pile_train_cleaned_v2_vocab.txt', 'r') as f:
  chars = f.read()

stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}

# encoder: string to int
encode = lambda s: [stoi[c] for c in s if c in stoi]

# decoder: int to string
decode = lambda l: ''.join([itos[i] for i in l if i in itos])

vocab_size = len(chars)

# single head
class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)

    wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)

    v = self.value(x)
    out = wei @ v
    return out

# multi-head
class MultiHeadAttention(nn.Module):

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):

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
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class GPTLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T)) #, device=device
    x = tok_emb + pos_emb 
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)

    if targets is None:
        loss = None
    else:
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    last_char = ''
    iter = 0 

    while last_char not in VALID_END_CHARS or iter < max_new_tokens:
        idx_cond = idx[:, -block_size:]
        logits, loss = self(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        iter += 1
        decode_idx = decode(idx[0].tolist())
        last_char = decode_idx[-1]

        yield decode_idx

    return idx

model = GPTLanguageModel()

checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

# Generator that returns output for chatbox 
def generate_response(message, history):
    context = torch.tensor(encode(message), dtype=torch.long)

    for value in model.generate(context.unsqueeze(0), max_new_tokens=MIN_CHAR_OUTPUT):  
        yield value

    return ''

# Create chatbox UI
demo = gr.ChatInterface(generate_response,    

    theme="soft",

    chatbot=gr.Chatbot(height=755),
    textbox=gr.Textbox(placeholder=TEXTBOX_PLACEHOLDER, container=False, scale=7),

    title=INTERFACE_TITLE,
    description=INTERFACE_DESCRIPTION,
    examples=INTERFACE_EXAMPLE_ENTRIES,

    cache_examples=False,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
    ).queue()

demo.launch()