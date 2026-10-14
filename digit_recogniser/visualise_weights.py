
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load your trained neural network
model_path = Path(__file__).with_name("digit_model.npz")

with np.load(model_path) as model:
    W1 = model["W1"]

# W1 contains 784 weights for each of 128 hidden neurons
print("Weight matrix shape:", W1.shape)

# Display all 128 neurons in a grid
fig, axes = plt.subplots(8, 16, figsize=(16, 9))

fig.suptitle(
    "What Has My Neural Network Learned?",
    fontsize=18,
    fontweight="bold"
)

# Use the same colour scale across all neurons
limit = np.max(np.abs(W1))

for i, ax in enumerate(axes.flat):

    # Extract 784 weights belonging to neuron i
    neuron_weights = W1[:, i]

    # Rearrange into a 28 x 28 image
    weight_image = neuron_weights.reshape(28, 28)

    ax.imshow(
        weight_image,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit
    )

    ax.set_title(f"N{i + 1}", fontsize=7)
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    model_path.with_name("learned_weights.png"),
    dpi=200
)
plt.show()
