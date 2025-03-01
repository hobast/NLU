from transformers import GPT2Model
import torch

# Load the model
model = GPT2Model.from_pretrained("gpt2")

# Extract d_k from the first attention layer
first_attention_layer = model.h[0].attn
dk = first_attention_layer.head_dim

print("d_k (head_dim):", dk)
print("Scaling factor sqrt(d_k):", torch.sqrt(torch.tensor(dk, dtype=torch.float32)))
print ("#################################################################")
print (model)
print ("#################################################################")
for i, layer in enumerate(model.h):
    print(f"Layer {i} Attention:")
    print(layer.attn)