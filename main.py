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

  def forward(self,x,past_k=None,past_v=None):
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

    if past_k is not None:
        K = torch.cat([past_k,K],dim=2)

    if past_v is not None:
        V = torch.cat([past_v,V],dim=2)

    scores = (
        Q @ K.transpose(-2, -1)
    ) / math.sqrt(head_dim)


    if past_k is None:
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


    return out,K,V

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


  def forward(self,x,past_k=None,past_v=None):

    h = self.ln1.forward(x)

    attn,K,V = self.self_attention.forward(h,past_k,past_v)

    x = x + attn

    h = self.ln2.forward(x)

    mlp_out = self.mlp.forward(h)

    x = x + mlp_out

    return x,K,V

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

    self.lm_head = model.transformer.wte.weight.T

  # texts -> tokens
  def tokenize(self,text):
    tokens = self.tokenizer.encode(text)
    return tokens

  # tokens -> texts
  def detokenize(self,tokens):
    text = self.tokenizer.decode(tokens)
    return text

  # token and positional embeddings
  def embeddings(
    self,
    tokens,
    position_offset=0
  ):

    if isinstance(tokens, list):
        tokens = torch.tensor([tokens])

    wte = self.model.transformer.wte.weight
    wpe = self.model.transformer.wpe.weight

    token_embeddings = wte[tokens]

    T = tokens.size(1)

    positions = torch.arange(
        position_offset,
        position_offset + T,
        device=tokens.device
    )

    positional_embeddings = wpe[positions]

    return token_embeddings + positional_embeddings

  def transformer(self, x,past_keys=None,past_values=None):

    if past_keys is None:
        past_keys = [None]* len(self.blocks)

    if past_values is None:
       past_values = [None]* len(self.blocks)

    new_keys= []
    new_values= []

    for i,block in enumerate(self.blocks):
        x,k,v = block.forward(x,past_keys[i],past_values[i])
        new_keys.append(k)
        new_values.append(v)

    x = self.ln_f.forward(x)

    return x,new_keys,new_values

  def sample(self,next_token_logits):
    k = 50
    temperature = 0.8

    values, indices = torch.topk(
        next_token_logits,
        k
    )

    probs = torch.softmax(
        values / temperature,
        dim=-1
    )

    sample_idx = torch.multinomial(
        probs,
        1
    )

    next_token = torch.gather(
        indices,
        1,
        sample_idx
    )

    return next_token



  @torch.no_grad()
  def generate(self, prompt,max_new_token=100):

        # prompt -> tokens
        tokens = self.tokenize(prompt)

        # tokens -> embeddings
        x = self.embeddings(tokens)

        # embeddings -> context
        x,past_keys,past_values = self.transformer(x)

        # context -> logits
        logits = x @ self.lm_head

        for _ in range(max_new_token):

            next_token_logits = logits[:, -1, :]

            # logits -> next token
            next_token = self.sample(next_token_logits=next_token_logits)

            token_id = next_token.item()

            if token_id == self.tokenizer.eos_token_id:
                break

            tokens.append(token_id)

            # New Token
            token_tensor = torch.tensor(
                [[token_id]]
            )

            position_offset = len(tokens) - 1

            if position_offset >= 1024:
                break

            x = self.embeddings(
                token_tensor,
                position_offset
            )

            x,past_keys,past_values = self.transformer(
                x,
                past_keys,
                past_values
            )

            logits = x@ self.lm_head

        return self.detokenize(tokens)


def main():

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt = GPT2(tokenizer, model)

    prompt = input("> ")

    text = gpt.generate(
        prompt
    )

    print(text)


if __name__ == "__main__":
    main()