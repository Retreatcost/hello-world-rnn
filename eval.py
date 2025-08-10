import torch
import torch.nn as nn
from safetensors.torch import load_model

# Hyperparameters
SEQ_LENGTH = 3
HIDDEN_SIZE = 32

# 1. Prepare Data
text = "hello world"
chars = sorted(set(text))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


# 3. Define Model
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        _, hidden = self.rnn(x)
        out = self.fc(hidden.squeeze(0))
        return out


model = CharRNN(len(chars), HIDDEN_SIZE)

load_model(model, filename="model.safetensors")


def generate_text(seed, length=5):
    model.eval()
    with torch.no_grad():
        # Start with seed sequence
        generated = seed

        for _ in range(length):
            # Convert last SEQ_LENGTH chars to tensor
            input_seq = torch.tensor(
                [char_to_idx[ch] for ch in generated[-SEQ_LENGTH:]], dtype=torch.long
            ).unsqueeze(0)

            # Predict next character
            output = model(input_seq)
            prob = torch.softmax(output, dim=-1)
            next_idx = torch.multinomial(prob, 1).item()
            generated += idx_to_char[next_idx]

        return generated


# Test predictions
print("Predictions from loaded model:")
print(f"Seed 'hel' → {generate_text('hel', 2)}")  # Should extend to "hello"
print(f"Seed 'wor' → {generate_text('wor', 2)}")  # Should extend to "world"
