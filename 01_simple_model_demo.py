import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seeds for reproducible results
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("  STEP 1: GENERATE TRAINING & TESTING DATA")
print("=" * 60)
# We create 1,000 random examples of (dx, dy) distance to a target
# Rules:
# dx > 0 and |dx| > |dy| => East  (Label 1)
# dx < 0 and |dx| > |dy| => West  (Label 3)
# dy > 0 and |dy| >= |dx| => North (Label 0)
# dy < 0 and |dy| >= |dx| => South (Label 2)

def generate_data(num_samples=1000):
    X = np.random.uniform(-10.0, 10.0, size=(num_samples, 2)).astype(np.float32)
    y = []
    for dx, dy in X:
        if abs(dy) >= abs(dx):
            y.append(0 if dy > 0 else 2) # 0: North, 2: South
        else:
            y.append(1 if dx > 0 else 3) # 1: East, 3: West
    return torch.tensor(X), torch.tensor(y, dtype=torch.long)

X_data, y_data = generate_data(1000)

# Split into 80% Train (800 samples) and 20% Test (200 samples)
X_train, X_test = X_data[:800], X_data[800:]
y_train, y_test = y_data[:800], y_data[800:]

print(f"Generated {len(X_train)} training examples and {len(X_test)} test examples.")
print(f"Sample Input (dx, dy): {X_train[0].tolist()} -> Target Action: {y_train[0].item()} (0=North, 1=East, 2=South, 3=West)")

print("\n" + "=" * 60)
print("  STEP 2: DEFINE THE NEURAL NETWORK MODEL")
print("=" * 60)

class SimpleNavigator(nn.Module):
    def __init__(self):
        super(SimpleNavigator, self).__init__()
        # Input: 2 numbers (dx, dy)
        # Hidden Layer: 16 neurons with ReLU activation
        # Output: 4 numbers (scores for North, East, South, West)
        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        # Data flow through layers
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits

model = SimpleNavigator()
print("Model Architecture:")
print(model)

print("\n" + "=" * 60)
print("  STEP 3: SETUP LOSS FUNCTION & OPTIMIZER")
print("=" * 60)
# Loss function calculates HOW WRONG the model's prediction is
criterion = nn.CrossEntropyLoss()

# Optimizer adjusts the weights using gradients to reduce loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("\n" + "=" * 60)
print("  STEP 4: TRAINING LOOP (LEARNING PHASE)")
print("=" * 60)

epochs = 30
for epoch in range(1, epochs + 1):
    model.train() # Set model to training mode
    
    # 1. Forward pass: compute predicted outputs
    predictions = model(X_train)
    
    # 2. Calculate loss
    loss = criterion(predictions, y_train)
    
    # 3. Clear old gradients
    optimizer.zero_grad()
    
    # 4. Backward pass: compute gradients (how to change each weight)
    loss.backward()
    
    # 5. Optimizer step: update weights
    optimizer.step()
    
    # Calculate training accuracy
    predicted_classes = torch.argmax(predictions, dim=1)
    accuracy = (predicted_classes == y_train).float().mean() * 100
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:2d}/{epochs} | Loss: {loss.item():.4f} | Training Accuracy: {accuracy:.2f}%")

print("\n" + "=" * 60)
print("  STEP 5: TESTING & EVALUATION PHASE")
print("=" * 60)
model.eval() # Set model to evaluation mode (no weight updates)

with torch.no_grad(): # Don't compute gradients during testing
    test_predictions = model(X_test)
    test_loss = criterion(test_predictions, y_test)
    test_predicted_classes = torch.argmax(test_predictions, dim=1)
    test_accuracy = (test_predicted_classes == y_test).float().mean() * 100

print(f"Test Loss: {test_loss.item():.4f}")
print(f"Final Test Accuracy: {test_accuracy:.2f}%")

print("\n" + "=" * 60)
print("  STEP 6: TRYING OUT THE TRAINED MODEL ON NEW SAMPLES")
print("=" * 60)
direction_names = ["North (0)", "East (1)", "South (2)", "West (3)"]

sample_inputs = torch.tensor([
    [0.0, 5.0],   # Food is directly North
    [7.0, 0.0],   # Food is directly East
    [0.0, -8.0],  # Food is directly South
    [-4.0, 0.0],  # Food is directly West
], dtype=torch.float32)

with torch.no_grad():
    sample_outputs = model(sample_inputs)
    predicted_actions = torch.argmax(sample_outputs, dim=1)

for idx, input_vec in enumerate(sample_inputs):
    dx, dy = input_vec.tolist()
    chosen_action = direction_names[predicted_actions[idx].item()]
    print(f"Target at (dx={dx:+.1f}, dy={dy:+.1f}) -> Model chose: {chosen_action}")

print("\nTraining & Testing complete!")
