import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

directions_name = ["North_2", "East_0", "South_3", "West_1"]

def generateRandomInputs(num_samples = 1000):
    X = np.random.uniform(-10.0, 10.0, size=(num_samples, 4)).astype(np.float32)
    y = []
    for x_food, y_food, x_enemy, y_enemy in X:
        if abs(y_food) >= abs(x_food):
            y.append(2 if y_food > 0 else 3) # South_3, North_2
        else:
            y.append(0 if x_food > 0 else 1) # East_0, West_1
    return torch.tensor(X), torch.tensor(y, dtype=torch.long)

X_train, y_train = generateRandomInputs(1000)
print(X_train[0])
print(y_train[0])