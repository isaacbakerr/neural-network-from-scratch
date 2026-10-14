
import numpy as np
import matplotlib.pyplot as plt

# 1. TRAINING DATA
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 2. INITIALISE NETWORK
rng = np.random.default_rng(42)

W1 = rng.normal(0, 1, (2, 4))
b1 = np.zeros((1, 4))

W2 = rng.normal(0, 1, (4, 1))
b2 = np.zeros((1, 1))

learning_rate = 1.0
epochs = 20000

loss_history = []


# 3. TRAIN THE NETWORK
for epoch in range(epochs):

    # Forward propagation
    hidden = sigmoid(X @ W1 + b1)
    predictions = sigmoid(hidden @ W2 + b2)

    # Calculate loss
    loss = -np.mean(
        y * np.log(predictions + 1e-9)
        + (1 - y) * np.log(1 - predictions + 1e-9)
    )

    loss_history.append(loss)

    # Backpropagation
    output_error = (predictions - y) / len(X)

    dW2 = hidden.T @ output_error
    db2 = np.sum(output_error, axis=0, keepdims=True)

    hidden_error = (
        (output_error @ W2.T)
        * hidden * (1 - hidden)
    )

    dW1 = X.T @ hidden_error
    db1 = np.sum(hidden_error, axis=0, keepdims=True)

    # Gradient descent
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 2000 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.6f}")


# 4. REUSABLE PREDICTION FUNCTION
def predict(inputs):
    hidden = sigmoid(inputs @ W1 + b1)
    return sigmoid(hidden @ W2 + b2)


# 5. CALCULATE ACCURACY
final_predictions = predict(X)

rounded = (final_predictions >= 0.5).astype(int)
accuracy = np.mean(rounded == y) * 100

print(f"\nTraining accuracy: {accuracy:.1f}%")
print("Predictions:")
print(final_predictions.round(4))


# 6. SAVE THE TRAINED MODEL
np.savez(
    "neural_network_model.npz",
    W1=W1,
    b1=b1,
    W2=W2,
    b2=b2
)

print("\nModel saved successfully!")


# 7. VISUALISE TRAINING
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label="Training Loss")

plt.title("Neural Network Learning Progress")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("training_graph.png")
plt.show()


# 8. INTERACTIVE PREDICTIONS
print("\n--- Test Your Neural Network ---")

while True:
    user_input = input(
        "\nEnter two binary digits (e.g. 1 0), or q to quit: "
    ).strip()

    if user_input.lower() == "q":
        break

    values = user_input.split()

    if len(values) != 2 or any(
        value not in ("0", "1") for value in values
    ):
        print("Please enter exactly two digits: 0 or 1.")
        continue

    numbers = np.array(
        [[float(value) for value in values]]
    )

    result = predict(numbers)[0, 0]

    print(f"Raw prediction: {result:.6f}")
    print(f"Predicted XOR: {int(result >= 0.5)}")
