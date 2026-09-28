import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seeds for reproducible results
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("  EXERCISE 2: TACTICAL ANT NAVIGATOR (FOOD + ENEMY FLEEING)")
print("=" * 60)

# STEP 1: GENERATE DATA WITH 4 INPUT FEATURES & 5 ACTIONS
# Input Features: [dx_food, dy_food, dx_enemy, dy_enemy]
# Actions: 0=North, 1=East, 2=South, 3=West, 4=Stay

def generate_data(num_samples=1000):
    X = np.random.uniform(-10.0, 10.0, size=(num_samples, 4)).astype(np.float32)
    y = []
    
    for dx_food, dy_food, dx_enemy, dy_enemy in X:
        enemy_dist = np.sqrt(dx_enemy**2 + dy_enemy**2)
        
        # RULE 1: If Enemy is dangerously close (distance <= 2.0), FLEE from Enemy!
        if enemy_dist <= 2.0:
            if abs(dy_enemy) >= abs(dx_enemy):
                y.append(2 if dy_enemy > 0 else 0) # Enemy North -> Flee South (2), Enemy South -> Flee North (0)
            else:
                y.append(3 if dx_enemy > 0 else 1) # Enemy East -> Flee West (3), Enemy West -> Flee East (1)
        
        # RULE 2: Otherwise, move toward Food
        elif abs(dy_food) >= abs(dx_food):
            y.append(0 if dy_food > 0 else 2) # 0: North, 2: South
        else:
            y.append(1 if dx_food > 0 else 3) # 1: East, 3: West
            
    return torch.tensor(X), torch.tensor(y, dtype=torch.long)

X_data, y_data = generate_data(1000)

# Split into 80% Train (800 samples) and 20% Test (200 samples)
X_train, X_test = X_data[:800], X_data[800:]
y_train, y_test = y_data[:800], y_data[800:]

print(f"Generated {len(X_train)} training examples and {len(X_test)} test examples.")

# STEP 2: NEURAL NETWORK ARCHITECTURE (4 INPUTS -> 32 HIDDEN -> 5 OUTPUTS)

class TacticalAntNavigator(nn.Module):
    def __init__(self):
        super(TacticalAntNavigator, self).__init__()
        self.fc1 = nn.Linear(4, 32) # 4 Input Features
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 5) # 5 Actions (North, East, South, West, Stay)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits

model = TacticalAntNavigator()
print("\nModel Architecture:")
print(model)

# STEP 3: LOSS FUNCTION & OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# STEP 4: TRAINING LOOP
print("\n" + "=" * 60)
print("  TRAINING LOOP")
print("=" * 60)

epochs = 30
for epoch in range(1, epochs + 1):
    model.train()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    predicted_classes = torch.argmax(predictions, dim=1)
    accuracy = (predicted_classes == y_train).float().mean() * 100
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:2d}/{epochs} | Loss: {loss.item():.4f} | Training Accuracy: {accuracy:.2f}%")

# STEP 5: EVALUATION PHASE
print("\n" + "=" * 60)
print("  EVALUATION PHASE")
print("=" * 60)
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = criterion(test_predictions, y_test)
    test_predicted_classes = torch.argmax(test_predictions, dim=1)
    test_accuracy = (test_predicted_classes == y_test).float().mean() * 100

print(f"Test Loss: {test_loss.item():.4f}")
print(f"Final Test Accuracy: {test_accuracy:.2f}%")

# STEP 6: TEST TACTICAL ANTS SCENARIOS
print("\n" + "=" * 60)
print("  TESTING TACTICAL ANTS SCENARIOS")
print("=" * 60)

action_labels = ["North (0)", "East (1)", "South (2)", "West (3)", "Stay (4)"]

# Test scenarios: [dx_food, dy_food, dx_enemy, dy_enemy]
scenarios = [
    ("A: Food North, Enemy Far Away",    [0.0, 5.0,  8.0, 8.0]),  # Should go NORTH (0)
    ("B: Food North, BUT Enemy Near North", [0.0, 5.0,  0.0, 1.5]),  # Should FLEE SOUTH (2)
    ("C: Food East,  BUT Enemy Near West",  [7.0, 0.0, -1.0, 0.0]),  # Should FLEE EAST (1)
]

for title, input_vector in scenarios:
    inp_tensor = torch.tensor([input_vector], dtype=torch.float32)
    with torch.no_grad():
        out = model(inp_tensor)
        pred = torch.argmax(out, dim=1).item()
    print(f"Scenario {title}")
    print(f" -> Input: {input_vector}")
    print(f" -> Action Chosen: {action_labels[pred]}\n")
