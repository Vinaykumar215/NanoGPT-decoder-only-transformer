import torch
import torch.nn as nn
from torch.nn import functional as F

# =====================================================
# HYPERPARAMETERS
# =====================================================
batch_size = 32
block_size = 8
max_iters = 5000
learning_rate = 1e-3
eval_interval = 500
eval_iters = 200
train_split = 0.9
seed = 1337
device='cuda' if torch.cuda.is_available() else 'cpu'
generate_tokens = 500
vocab_size=65
n_embd=32
head_size=16
torch.manual_seed(seed)

# =====================================================
# LOAD DATA
# =====================================================
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# =====================================================
# TOKENIZER
# =====================================================
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# =====================================================
# DATASET
# =====================================================
data = torch.tensor(encode(text), dtype=torch.long)

n = int(train_split * len(data))
train_data = data[:n]
val_data = data[n:]


# =====================================================
# BATCH GENERATOR
# =====================================================
def get_batch(split):
    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y=get_batch(split)
            logits,loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out


# =====================================================
# MODEL
# =====================================================
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        #each token directl reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd) #each char 65 chars instead of boring number as 1 it has stream of 32 number reprsenting some numbers ex[0.12,0.53,-0.16....0.25]
        self.pos_embdedding=nn.Embedding(block_size,n_embd) #and each char in the block size know their postions from 0 to T-1
        self.lm_head=nn.Linear(n_embd,vocab_size) #just a linear layer not doing much

    def forward(self, idx, targets=None):
        B,T=idx.shape
        tok_embd= self.token_embedding_table(idx) #B,T,C=n+embd
        B, T, C = tok_embd.shape
        pos_idx = torch.arange(T, device=device)
        pos_embd = self.pos_embdedding(pos_idx) #(T,C)
        x = tok_embd + pos_embd.unsqueeze(0) #B,T,C: broadcast position embeddings across batch
        logits=self.lm_head(x) #B,T,C=vocab_size      m*nXn*m=m*m
        

        if targets is None:
            loss = None
        else:

            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self,idx,max_new_tokens): #first we start with input zero which is new line char see next box
        for _ in range(max_new_tokens): #for(int i=0;i<100;i++)
            idx_cond = idx[:, -block_size:]   # keep last 8 tokens

            logits, _ = self(idx_cond) #gets the value from forward
            logits=logits[:,-1,:] #this just see the previous char to predict the curr one
            probs=F.softmax(logits,dim=-1) #then we softmax them meaning assign probablitites for the all the char we have bcz intially logits is jut random scores given to every 65 of them
            idx_next=torch.multinomial(probs,num_samples=1) #radnomly picks e one
            idx=torch.cat((idx,idx_next),dim=1) #then we concatenate the input we have idx and the most probable next char our MODEL predicts
        return idx



# =====================================================
# CREATE MODEL
# =====================================================
model = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

# =====================================================
# TRAINING
# =====================================================
for iter in range(max_iters):

    if iter % eval_interval==0:
        losses=estimate_loss()
        print(f"step {iter}: train loss{losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

  

# =====================================================
# GENERATE TEXT
# =====================================================
context = torch.zeros((1, 1), dtype=torch.long,device=device)

generated = model.generate(
    context,
    max_new_tokens=generate_tokens
)

print("\n================ GENERATED TEXT ================\n")
print(decode(generated[0].tolist()))