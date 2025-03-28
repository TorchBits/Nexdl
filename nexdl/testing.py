from nexlib import nexdl as nx 
import nexlib.optim as optim
import nexlib.nn as nn 

nx.backend.random.seed(42)

X = nx.tensor(nx.backend.random.randn(100,5))
y = nx.tensor(nx.backend.random.randn(100,1))


model = nn.Linear(5,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
epochs = 100

for epoch in range(epochs):
    # Forward pass
    predictions = model(X)
    print(predictions.shape)
    loss = criterion(predictions, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Final Output
print("\nTraining Complete!")
print("Final Weights:", model.weight.data)
print("Final Bias:", model.bias.data)




