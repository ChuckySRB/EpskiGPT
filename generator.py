import torch
import torch.nn.functional as F
from gpt import GPT
from gpt import decode
from gpt import load_data

def generate(model, block_size, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        # get the predictions
        logits, loss = model(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generate_length = 2000
    model_path = "./model/model_checkpoint.pt"
    input_file = "./data/narodne_pesme.txt"
    _, _, _, itos, vocabulary_size = load_data(input_file)
    # Load and prepare the model
    model_dict = torch.load(model_path)
    model = GPT(vocabulary_size)
    model.load_state_dict(model_dict['model_state_dict'])
    model.to(device)



    # Generate text and save to file
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(generate(model, 256, context, max_new_tokens=generate_length)[0].tolist(), itos)

    # Save generated text and log to wandb
    output_file = 'output/generated_text.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(generated_text)

