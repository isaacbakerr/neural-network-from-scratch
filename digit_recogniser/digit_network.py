
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from urllib.request import urlretrieve

# 1. DOWNLOAD THE HANDWRITTEN DIGITS

DATA_FILE = Path(__file__).with_name("mnist.npz")

if not DATA_FILE.exists():
    print("Downloading handwritten digit dataset...")
    urlretrieve(
        "https://storage.googleapis.com/tensorflow/"
        "tf-keras-datasets/mnist.npz",
        DATA_FILE
    )

with np.load(DATA_FILE) as data:
    X_train = data["x_train"]
    y_train = data["y_train"]
    X_test = data["x_test"]
    y_test = data["y_test"]

# Start with a smaller subset for faster training
# Use the complete MNIST dataset
# 60,000 training images
# 10,000 test images

# No slicing required

# Convert images into 784 input values from 0 to 1
X_train = X_train.reshape(-1, 784).astype(np.float32) / 255
X_test = X_test.reshape(-1, 784).astype(np.float32) / 255

print("Training images:", len(X_train))
print("Testing images:", len(X_test))


# 2. ACTIVATION FUNCTIONS

def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# 3. CREATE THE NEURAL NETWORK

rng = np.random.default_rng(42)

W1 = (
    rng.standard_normal((784, 128)).astype(np.float32)
    * np.sqrt(2 / 784)
)
b1 = np.zeros((1, 128), dtype=np.float32)

W2 = (
    rng.standard_normal((128, 10)).astype(np.float32)
    * np.sqrt(2 / 128)
)
b2 = np.zeros((1, 10), dtype=np.float32)

learning_rate = 0.05
epochs = 15
batch_size = 128

loss_history = []
accuracy_history = []


# 4. FORWARD PROPAGATION

def forward(X):
    hidden = relu(X @ W1 + b1)
    predictions = softmax(hidden @ W2 + b2)
    return hidden, predictions


# 5. TRAIN THE NETWORK

print("\nStarting training...")

for epoch in range(epochs):

    indices = rng.permutation(len(X_train))

    for start in range(0, len(X_train), batch_size):

        batch_indices = indices[start:start + batch_size]

        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        # Forward propagation
        hidden, predictions = forward(X_batch)

        # Backpropagation
        output_error = predictions.copy()
        output_error[np.arange(len(y_batch)), y_batch] -= 1
        output_error /= len(y_batch)

        dW2 = hidden.T @ output_error
        db2 = np.sum(output_error, axis=0, keepdims=True)

        hidden_error = (
            (output_error @ W2.T)
            * (hidden > 0)
        )

        dW1 = X_batch.T @ hidden_error
        db1 = np.sum(hidden_error, axis=0, keepdims=True)

        # Gradient descent
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    # Measure training progress
    _, predictions = forward(X_train)

    loss = -np.mean(
        np.log(
            np.clip(
                predictions[np.arange(len(y_train)), y_train],
                1e-9,
                1.0
            )
        )
    )

    accuracy = np.mean(
        np.argmax(predictions, axis=1) == y_train
    ) * 100

    loss_history.append(loss)
    accuracy_history.append(accuracy)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {loss:.4f} | "
        f"Training Accuracy: {accuracy:.2f}%"
    )


# 6. TEST ON UNSEEN IMAGES

_, test_predictions = forward(X_test)

test_accuracy = np.mean(
    np.argmax(test_predictions, axis=1) == y_test
) * 100

print(f"\nTEST ACCURACY: {test_accuracy:.2f}%")


# 7. SAVE THE NEURAL NETWORK

model_path = DATA_FILE.with_name("digit_model.npz")

np.savez(model_path, W1=W1, b1=b1, W2=W2, b2=b2)
print("Trained network saved!")


# 8. GRAPH THE TRAINING

plt.figure(figsize=(9, 5))
plt.plot(
    range(1, epochs + 1),
    accuracy_history,
    marker="o"
)

plt.title("Neural Network Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(DATA_FILE.with_name("digit_training.png"))
plt.show()


# 9. SHOW PREDICTIONS ON HANDWRITTEN DIGITS

_, predictions = forward(X_test[:10])

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    image = X_test[i].reshape(28, 28)
    predicted_digit = np.argmax(predictions[i])
    actual_digit = y_test[i]

    ax.imshow(image, cmap="gray")
    ax.set_title(
        f"Predicted: {predicted_digit}\n"
        f"Actual: {actual_digit}"
    )
    ax.axis("off")

plt.suptitle("My Neural Network's Predictions")
plt.tight_layout()
plt.show()
