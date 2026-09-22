from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
import torch
import math
import torch.nn.functional as F

class LayerNorm:
  def __init__(self,weight,bias,eps=1e-5):
    self.weight= weight
    self.bias= bias
    self.eps= eps

  def forward(self,x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_hat = (x - mean) / torch.sqrt(var + self.eps)
    return self.weight * x_hat + self.bias


class SelfAttention:
  def __init__(self,c_attn_weight,c_attn_bias,c_proj_weight,c_proj_bias,n_head=12):
     self.c_attn_weight= c_attn_weight
     self.c_attn_bias= c_attn_bias
     self.c_proj_weight= c_proj_weight
     self.c_proj_bias= c_proj_bias
     self.n_head= n_head

  def forward(self,x):
    B, T, C = x.shape
    
    qkv = x @ self.c_attn_weight + self.c_attn_bias

    Q, K, V = torch.split(
        qkv,
        C,
        dim=-1
    )

    head_dim = C // self.n_head

    Q = Q.view(B, T, self.n_head, head_dim).transpose(1, 2)
    K = K.view(B, T, self.n_head, head_dim).transpose(1, 2)
    V = V.view(B, T, self.n_head, head_dim).transpose(1, 2)

    scores = (
        Q @ K.transpose(-2, -1)
    ) / math.sqrt(head_dim)

    mask = torch.tril(
        torch.ones(
            T,
            T,
            device=x.device
        )
    )

    scores = scores.masked_fill(
        mask == 0,
        float("-inf")
    )

    weights = torch.softmax(
        scores,
        dim=-1
    )

    context = weights @ V

    context = (
        context
        .transpose(1, 2)
        .contiguous()
        .view(B, T, C)
    )

    out = (
        context @ self.c_proj_weight
        + self.c_proj_bias
    )

    return out

class MLP:
  def __init__(self,fc_weight,fc_bias,proj_weight,proj_bias):
    self.fc_weight= fc_weight
    self.fc_bias= fc_bias
    self.proj_weight= proj_weight
    self.proj_bias= proj_bias

  def forward(self,x):
    # 768 -> 3072
    h = x @ self.fc_weight + self.fc_bias

    # GELU activation
    h = F.gelu(h)

    # 3072 -> 768
    out = h @ self.proj_weight + self.proj_bias

    return out


class TransformerBlock:
  def __init__(self,block):
     self.ln1 = LayerNorm(
        block.ln_1.weight,
        block.ln_1.bias
     )

     self.ln2 = LayerNorm(
      block.ln_2.weight,
      block.ln_2.bias
    )

     self.self_attention = SelfAttention(
      c_attn_weight= block.attn.c_attn.weight,
      c_attn_bias= block.attn.c_attn.bias,
      c_proj_weight= block.attn.c_proj.weight,
      c_proj_bias= block.attn.c_proj.bias
    )

     self.mlp= MLP(
      fc_weight= block.mlp.c_fc.weight,
      fc_bias= block.mlp.c_fc.bias,
      proj_weight= block.mlp.c_proj.weight,
      proj_bias= block.mlp.c_proj.bias
  )


  def forward(self,x):

    h = self.ln1.forward(x)

    attn = self.self_attention.forward(h)

    x = x + attn

    h = self.ln2.forward(x)

    mlp_out = self.mlp.forward(h)

    x = x + mlp_out

    return x

class GPT2:
  def __init__(self,tokenizer,model):
    self.tokenizer= tokenizer
    self.model= model

    self.blocks = [
            TransformerBlock(block)
            for block in model.transformer.h
        ]

    self.ln_f = LayerNorm(
            model.transformer.ln_f.weight,
            model.transformer.ln_f.bias
        )

  # texts -> tokens
  def tokenize(self,text):
    tokens = self.tokenizer.encode(text)
    return tokens

  # tokens -> texts
  def detokenize(self,tokens):
    text = self.tokenizer.decode(tokens)
    return text

  # token and positional embeddings
  def embeddings(self,tokens):
    if isinstance(tokens, list):
          tokens = torch.tensor([tokens])
    wte = self.model.transformer.wte.weight
    wpe = self.model.transformer.wpe.weight

    token_embeddings = wte[tokens]

    T = tokens.size(1)
    positions = torch.arange(T, device=tokens.device)

    positional_embeddings = wpe[positions]
    return token_embeddings + positional_embeddings

  def transformer(self, x):

    for block in self.blocks:
        x = block.forward(x)

    x = self.ln_f.forward(x)

    return x

  def generate(self, prompt, max_new_tokens=10):

        tokens = self.tokenize(prompt)

        for _ in range(max_new_tokens):

            x = self.embeddings(tokens)

            x = self.transformer(x)

            logits = x @ self.model.transformer.wte.weight.T

            next_token_logits = logits[:, -1, :]

            probs = torch.softmax(
                next_token_logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1
            ).item()

            tokens.append(next_token)

        return self.detokenize(tokens)


def main():

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    gpt = GPT2(tokenizer, model)

    prompt = input("> ")

    text = gpt.generate(
        prompt,
        max_new_tokens=20
    )

    print(text)


if __name__ == "__main__":
    main()