
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageTk

# ----------------------------------
# 1. LOAD YOUR TRAINED NETWORK
# ----------------------------------

MODEL_PATH = Path(__file__).with_name("digit_model.npz")

with np.load(MODEL_PATH) as model:
    W1 = model["W1"]
    b1 = model["b1"]
    W2 = model["W2"]
    b2 = model["b2"]


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward(image):
    # First layer: 784 pixels -> 128 neurons
    hidden = relu(image @ W1 + b1)

    # Second layer: 128 neurons -> 10 digits
    output = softmax(hidden @ W2 + b2)

    return hidden[0], output[0]


# ----------------------------------
# 2. APPLICATION WINDOW
# ----------------------------------

BG = "#171923"
PANEL = "#242938"
TEAL = "#2DD4BF"
WHITE = "#FFFFFF"

root = tk.Tk()
root.title("NeuralNet | Inside the Neural Network")
root.geometry("1050x690")
root.configure(bg=BG)

title = tk.Label(
    root,
    text="INSIDE THE NEURAL NETWORK",
    font=("Arial", 22, "bold"),
    fg=WHITE,
    bg=BG
)
title.pack(pady=20)

workspace = tk.Frame(root, bg=BG)
workspace.pack(padx=20, fill="both", expand=True)

# Three columns
left = tk.Frame(workspace, bg=PANEL, padx=15, pady=15)
middle = tk.Frame(workspace, bg=PANEL, padx=15, pady=15)
right = tk.Frame(workspace, bg=PANEL, padx=15, pady=15)

left.grid(row=0, column=0, padx=8, sticky="n")
middle.grid(row=0, column=1, padx=8, sticky="n")
right.grid(row=0, column=2, padx=8, sticky="n")

workspace.grid_columnconfigure(0, weight=1)
workspace.grid_columnconfigure(1, weight=1)
workspace.grid_columnconfigure(2, weight=1)


def heading(parent, text):
    return tk.Label(
        parent,
        text=text,
        font=("Arial", 14, "bold"),
        fg=WHITE,
        bg=PANEL
    )


# ----------------------------------
# 3. DRAWING CANVAS
# ----------------------------------

heading(left, "INPUT LAYER").pack(pady=10)

canvas = tk.Canvas(
    left,
    width=280,
    height=280,
    bg="black",
    highlightthickness=0,
    cursor="cross"
)
canvas.pack(pady=10)

drawing = Image.new("L", (280, 280), 0)
pen = ImageDraw.Draw(drawing)

last_position = None
BRUSH = 20


def paint(event):
    global last_position

    x, y = event.x, event.y

    if last_position is not None:
        px, py = last_position

        canvas.create_line(
            px, py, x, y,
            fill=WHITE,
            width=BRUSH,
            capstyle=tk.ROUND
        )

        pen.line(
            (px, py, x, y),
            fill=255,
            width=BRUSH
        )

    r = BRUSH // 2

    canvas.create_oval(
        x-r, y-r, x+r, y+r,
        fill=WHITE,
        outline=WHITE
    )

    pen.ellipse(
        (x-r, y-r, x+r, y+r),
        fill=255
    )

    last_position = (x, y)


def stop_paint(event):
    global last_position
    last_position = None


canvas.bind("<Button-1>", paint)
canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", stop_paint)

heading(left, "WHAT THE AI SEES").pack(pady=(15, 5))

preview = tk.Label(left, bg="black")
preview.pack(pady=5)


# ----------------------------------
# 4. HIDDEN NEURON VISUALISATION
# ----------------------------------

heading(middle, "HIDDEN LAYER").pack(pady=10)

tk.Label(
    middle,
    text="128 neuron activations",
    fg="#9CA3AF",
    bg=PANEL
).pack()

neuron_canvas = tk.Canvas(
    middle,
    width=280,
    height=280,
    bg="#101827",
    highlightthickness=0
)
neuron_canvas.pack(pady=12)

neuron_items = []

for i in range(128):
    row = i // 8
    col = i % 8

    x = 25 + col * 32
    y = 12 + row * 17

    item = neuron_canvas.create_rectangle(
        x, y, x+23, y+12,
        fill="#293241",
        outline=""
    )

    neuron_items.append(item)

activation_label = tk.Label(
    middle,
    text="Waiting for input...",
    fg="#9CA3AF",
    bg=PANEL
)
activation_label.pack(pady=10)


# ----------------------------------
# 5. OUTPUT PROBABILITY BARS
# ----------------------------------

heading(right, "OUTPUT LAYER").pack(pady=10)

tk.Label(
    right,
    text="Digit probability scores",
    fg="#9CA3AF",
    bg=PANEL
).pack()

output_canvas = tk.Canvas(
    right,
    width=280,
    height=280,
    bg=PANEL,
    highlightthickness=0
)
output_canvas.pack(pady=10)

result_label = tk.Label(
    right,
    text="Prediction: ?",
    font=("Arial", 22, "bold"),
    fg=TEAL,
    bg=PANEL
)
result_label.pack(pady=15)


# ----------------------------------
# 6. IMAGE PREPROCESSING
# ----------------------------------

def prepare_image():
    bbox = drawing.getbbox()

    if bbox is None:
        return None, None

    digit = drawing.crop(bbox)

    width, height = digit.size
    scale = 20 / max(width, height)

    new_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale))
    )

    digit = digit.resize(
        new_size,
        Image.Resampling.LANCZOS
    )

    final = Image.new("L", (28, 28), 0)

    x = (28 - new_size[0]) // 2
    y = (28 - new_size[1]) // 2

    final.paste(digit, (x, y))

    # Centre using pixel intensity
    pixels = np.asarray(final, dtype=np.float32)
    total = pixels.sum()

    if total > 0:
        rows, cols = np.indices(pixels.shape)

        cy = (rows * pixels).sum() / total
        cx = (cols * pixels).sum() / total

        dx = round(13.5 - cx)
        dy = round(13.5 - cy)

        shifted = Image.new("L", (28, 28), 0)
        shifted.paste(final, (dx, dy))
        final = shifted

    array = np.asarray(
        final, dtype=np.float32
    ) / 255.0

    return array.reshape(1, 784), final


# ----------------------------------
# 7. UPDATE THE VISUALISATIONS
# ----------------------------------

def update_neurons(activations):

    # Scale activations for display only.
    maximum = max(float(np.max(activations)), 1e-9)

    for i, activation in enumerate(activations):
        strength = float(activation) / maximum

        # Dark neurons are inactive.
        # Bright teal neurons are strongly active.
        red = int(20 + 25 * strength)
        green = int(35 + 177 * strength)
        blue = int(45 + 146 * strength)

        colour = f"#{red:02x}{green:02x}{blue:02x}"

        neuron_canvas.itemconfig(
            neuron_items[i],
            fill=colour
        )

    active = np.sum(activations > 0)

    activation_label.config(
        text=f"Active neurons: {active}/128"
    )


def update_probabilities(probabilities):

    output_canvas.delete("all")

    prediction = int(np.argmax(probabilities))

    for digit, probability in enumerate(probabilities):

        y = 10 + digit * 27

        output_canvas.create_text(
            12, y+9,
            text=str(digit),
            fill=WHITE,
            font=("Arial", 11, "bold")
        )

        # Background bar
        output_canvas.create_rectangle(
            35, y,
            225, y+18,
            fill="#374151",
            outline=""
        )

        width = float(probability) * 190

        colour = TEAL if digit == prediction else "#64748B"

        output_canvas.create_rectangle(
            35, y,
            35 + width, y+18,
            fill=colour,
            outline=""
        )

        output_canvas.create_text(
            250, y+9,
            text=f"{probability*100:.1f}%",
            fill=WHITE,
            font=("Arial", 9)
        )

    confidence = float(probabilities[prediction]) * 100

    result_label.config(
        text=f"Prediction: {prediction}\n{confidence:.1f}%"
    )


# ----------------------------------
# 8. MAKE A PREDICTION
# ----------------------------------

def classify():
    image_array, processed_image = prepare_image()

    if image_array is None:
        result_label.config(text="Draw a digit!")
        return

    # Feed the drawing through YOUR network
    activations, probabilities = forward(image_array)

    # Display 28x28 input enlarged
    enlarged = processed_image.resize(
        (140, 140),
        Image.Resampling.NEAREST
    )

    photo = ImageTk.PhotoImage(enlarged)
    preview.config(image=photo)
    preview.image = photo

    # Update hidden neurons
    update_neurons(activations)

    # Update output probabilities
    update_probabilities(probabilities)


def clear():
    global drawing, pen, last_position

    canvas.delete("all")

    drawing = Image.new("L", (280, 280), 0)
    pen = ImageDraw.Draw(drawing)
    last_position = None

    preview.config(image="")
    preview.image = None

    for item in neuron_items:
        neuron_canvas.itemconfig(
            item,
            fill="#293241"
        )

    output_canvas.delete("all")
    result_label.config(text="Prediction: ?")
    activation_label.config(text="Waiting for input...")


# ----------------------------------
# 9. CONTROL BUTTONS
# ----------------------------------

controls = tk.Frame(root, bg=BG)
controls.pack(pady=18)

tk.Button(
    controls,
    text="PREDICT",
    command=classify,
    font=("Arial", 13, "bold"),
    bg="#0284C7",
    fg=WHITE,
    width=16
).grid(row=0, column=0, padx=10)

tk.Button(
    controls,
    text="CLEAR",
    command=clear,
    font=("Arial", 13, "bold"),
    bg="#374151",
    fg=WHITE,
    width=16
).grid(row=0, column=1, padx=10)

root.mainloop()
