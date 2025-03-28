import torch
import torch.nn as nn
import torch.optim as optim

# Simulated Dataset
torch.manual_seed(42)  # For reproducibility
X = torch.randn(100, 5)  # 100 samples, 5 features
y = torch.randn(100, 1)  # 100 target values (regression)

# Simple Linear Model
model = nn.Linear(5, 1)  # 5 input features → 1 output
print(model.bias.shape)
# Loss & Optimizer
criterion = nn.MSELoss()  # Mean Squared Error (Regression)

optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Training Loop
# epochs = 100
# for epoch in range(epochs):
#     # Forward pass
#     predictions = model(X)
#     print(predictions.shape)
#     loss = criterion(predictions, y)

#     # Backward pass
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # Print loss every 10 epochs
#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# # Final Output
# print("\nTraining Complete!")
# print("Final Weights:", model.weight.data)
# print("Final Bias:", model.bias.data)
