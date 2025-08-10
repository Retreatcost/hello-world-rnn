import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import save_file
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
SEQ_LENGTH = 3
HIDDEN_SIZE = 32
EPOCHS = 500

# Logging
writer = SummaryWriter()

# 1. Prepare Data
text = "hello world"
chars = sorted(set(text))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# 2. Create Training Data
inputs, targets = [], []
for i in range(len(text) - SEQ_LENGTH):
    inputs.append([char_to_idx[ch] for ch in text[i : i + SEQ_LENGTH]])
    targets.append(char_to_idx[text[i + SEQ_LENGTH]])

# Convert to PyTorch tensors
X = torch.tensor(inputs, dtype=torch.long)
y = torch.tensor(targets, dtype=torch.long)


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

# 4. Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. Train Model
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    writer.add_scalar("Loss/epoch", loss, epoch + 1)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}")

# Save and close writer
writer.flush()
writer.close()


# 6. Text Generation Function
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


output_path = "model.safetensors"
save_file(model.state_dict(), output_path)

# Test predictions
print("Predictions after training:")
print(f"Seed 'hel' → {generate_text('hel', 2)}")  # Should extend to "hello"
print(f"Seed 'wor' → {generate_text('wor', 2)}")  # Should extend to "world"
