"""NeuralNet Studio — a glossy Tkinter MNIST digit recogniser.

Requires numpy, pillow, and digit_model.npz in the same folder.
"""
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageTk, ImageFilter

BASE = Path(__file__).resolve().parent
MODEL = BASE / "digit_model.npz"
if not MODEL.exists():
    raise SystemExit(f"Missing model: {MODEL}\nCopy this script beside digit_model.npz")
with np.load(MODEL) as data:
    W1, b1, W2, b2 = (data[k] for k in ("W1", "b1", "W2", "b2"))

BG, CARD, INNER = "#08080C", "#15131A", "#211B23"
TEXT, MUTED, RED, ORANGE = "#FFF5F1", "#AA9CA1", "#F13243", "#FF9C3D"
SIZE, BRUSH = 320, 23

def rgb(h):
    return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))

def lerp(a, b, t):
    A, B = rgb(a), rgb(b)
    return "#" + "".join(f"{round(x + (y-x)*t):02x}" for x, y in zip(A, B))

def rounded_panel(w, h, radius=22, base=CARD, accent=False):
    """Draw a high-quality rounded card using Pillow (Tkinter has no rounded frames)."""
    scale = 2
    image = Image.new("RGBA", (w*scale, h*scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((2, 2, w*scale-3, h*scale-3), radius*scale, fill=base, outline="#34232D", width=2)
    if accent:
        for i in range(220):
            color = lerp(RED, ORANGE, i/219)
            draw.line((30*scale+i*scale, 3*scale, 31*scale+i*scale, 3*scale), fill=color, width=4)
    return ImageTk.PhotoImage(image.resize((w, h), Image.Resampling.LANCZOS))

def forward(x):
    h = np.maximum(0, x @ W1 + b1)
    z = h @ W2 + b2
    z -= z.max(axis=1, keepdims=True)
    ex = np.exp(z)
    return (ex / ex.sum(axis=1, keepdims=True))[0]

root = tk.Tk()
root.title("NEURALNET STUDIO  ·  Handwriting Intelligence")
root.geometry("1160x800")
root.minsize(1120, 780)
root.configure(bg=BG)

# UI surfaces use canvas rounded cards so no third-party GUI toolkit is required.
scene = tk.Canvas(root, bg=BG, highlightthickness=0, width=1160, height=800)
scene.pack(fill="both", expand=True)
refs = []

def text(x, y, value, size=12, color=TEXT, bold=False, anchor="nw", **kw):
    return scene.create_text(x, y, text=value, fill=color, anchor=anchor,
       font=("Segoe UI", size, "bold" if bold else "normal"), **kw)

def card(x, y, w, h, accent=False):
    p = rounded_panel(w, h, accent=accent)
    refs.append(p)
    scene.create_image(x, y, image=p, anchor="nw")

def gradient(x, y, width, height, start=RED, end=ORANGE, steps=120):
    for i in range(steps):
        a=x+width*i/steps; b=x+width*(i+1)/steps+1
        scene.create_rectangle(a, y, b, y+height, fill=lerp(start,end,i/max(steps-1,1)), outline="")

def action(x,y,w,h,label,command,accent=False):
    bg = "#A72C3D" if accent else "#29232C"
    item=scene.create_rectangle(x,y,x+w,y+h,fill=bg,outline="#5B313A" if accent else "#44343B",width=1)
    title=text(x+w/2,y+h/2,label,12,TEXT,True,anchor="center")
    for ident in (item,title):
        scene.tag_bind(ident,"<Button-1>",lambda _e: command())
        scene.tag_bind(ident,"<Enter>",lambda _e: root.configure(cursor="hand2"))
        scene.tag_bind(ident,"<Leave>",lambda _e: root.configure(cursor=""))

# Main design
text(48, 28, "N / N", 17, ORANGE, True)
text(125, 28, "NEURALNET", 19, TEXT, True)
text(125, 59, "STUDIO  /  FROM-SCRATCH AI", 9, MUTED)
text(1103, 37, "NUMPY ENGINE  •  MNIST", 10, ORANGE, True,anchor="ne")
gradient(48, 89, 1064, 3)
text(48, 113, "Make a mark. Watch intelligence respond.", 22, TEXT, True)
text(48, 153, "Draw any digit from 0 to 9. Your custom-trained neural network analyses it in real time.", 11, MUTED)
card(48, 198, 470, 515, accent=True)
card(538, 198, 574, 515, accent=True)
text(73, 221, "01  /  INPUT CANVAS", 12, ORANGE, True)
text(73, 248, "Draw a single digit", 15, TEXT, True)
text(73, 278, "White strokes on black · mouse or trackpad", 10, MUTED)
text(563, 221, "02  /  NEURAL OUTPUT", 12, ORANGE, True)
text(563, 250, "Prediction", 15, TEXT, True)
text(563, 381, "MODEL CONFIDENCE", 10, MUTED, True)
text(563, 448, "PROBABILITIES  ·  0–9", 10, MUTED, True)
text(48, 749, "NeuralNet Studio  •  784 → 128 → 10  •  ReLU / Softmax", 10, MUTED)
text(1112, 749, "Visual confidence is not guaranteed correctness", 9, MUTED, anchor="ne")

# Drawing surface embedded in canvas
brush_image = Image.new("L",(SIZE,SIZE),0)
pen = ImageDraw.Draw(brush_image)
last = None
pending = None
bars_after = None
bar_values = [0.0]*10
bar_targets = [0.0]*10
has_strokes=False

canvas = tk.Canvas(root, width=SIZE, height=SIZE, bg="#000000",
                   highlightbackground="#58404A", highlightthickness=1, cursor="crosshair")
scene.create_window(283, 474, window=canvas, width=SIZE+2, height=SIZE+2)
result = text(563, 275, "—", 67, TEXT, True)
confidence = text(563, 410, "Waiting for your drawing", 14, MUTED)
status = text(73, 661, "Draw to start live recognition", 10, MUTED)

# 28x28 live preview in top right
scene.create_rectangle(947,235,1086,374,fill="#000000",outline="#59414A",width=1)
text(1016, 380, "28 × 28 INPUT", 9, MUTED, anchor="n")
preview_item=scene.create_image(1016,304,anchor="center")
preview_photo=None

# Bars use canvas objects for inexpensive in-place resizing
bar_ids=[]
bar_text=[]
for digit in range(10):
    y=470+digit*21
    text(563,y,str(digit),10,TEXT,True)
    scene.create_rectangle(590,y+2,987,y+13,fill="#30262D",outline="")
    # Three gradient segments simulate red-to-orange fill with animated length
    ids=[]
    for j in range(18):
        ident=scene.create_rectangle(590,y+2,590,y+13,fill=lerp(RED,ORANGE,j/17),outline="")
        ids.append(ident)
    bar_ids.append(ids)
    bar_text.append(text(1090,y,"0.0%",10,MUTED,anchor="ne"))


def prepare_image():
    bbox=brush_image.getbbox()
    if not bbox: return None,None
    cropped=brush_image.crop(bbox)
    w,h=cropped.size
    factor=20/max(w,h)
    cropped=cropped.resize((max(1,round(w*factor)),max(1,round(h*factor))),Image.Resampling.LANCZOS)
    out=Image.new("L",(28,28),0)
    out.paste(cropped,((28-cropped.width)//2,(28-cropped.height)//2))
    a=np.asarray(out,dtype=np.float32)
    s=a.sum()
    if s:
        yy,xx=np.indices(a.shape)
        dx=round(13.5-(xx*a).sum()/s)
        dy=round(13.5-(yy*a).sum()/s)
        moved=Image.new("L",(28,28),0)
        moved.paste(out,(dx,dy))
        out=moved
    x=np.asarray(out,dtype=np.float32).reshape(1,784)/255.0
    return x,out


def refresh():
    global pending,preview_photo
    pending=None
    x,img=prepare_image()
    if x is None:
        scene.itemconfig(result,text="—")
        scene.itemconfig(confidence,text="Waiting for your drawing")
        return
    probs=forward(x)
    best=int(probs.argmax())
    scene.itemconfig(result,text=str(best))
    scene.itemconfig(confidence,text=f"{probs[best]*100:.1f}%  model confidence",fill=ORANGE)
    scene.itemconfig(status,text="Live inference active  •  Model loaded")
    preview_photo=ImageTk.PhotoImage(img.resize((132,132),Image.Resampling.NEAREST))
    scene.itemconfig(preview_item,image=preview_photo)
    for i in range(10):
        bar_targets[i]=float(probs[i])
    animate_bars()


def animate_bars():
    global bars_after
    bars_after=None
    more=False
    for i in range(10):
        target=bar_targets[i]
        bar_values[i]+=(target-bar_values[i])*0.27
        if abs(bar_values[i]-target)>0.001: more=True
        length=397*bar_values[i]
        for j, ident in enumerate(bar_ids[i]):
            xa=590+length*j/18
            xb=590+length*(j+1)/18
            scene.coords(ident,xa,472+i*21,xb,483+i*21)
        scene.itemconfig(bar_text[i],text=f"{bar_values[i]*100:.1f}%",
                         fill=ORANGE if i==int(np.argmax(bar_targets)) else MUTED)
    if more:
        bars_after=root.after(25,animate_bars)


def schedule_prediction():
    global pending
    if pending is not None: root.after_cancel(pending)
    pending=root.after(160,refresh)


def draw_at(event):
    global last,has_strokes
    x=max(0,min(SIZE-1,event.x));y=max(0,min(SIZE-1,event.y))
    rad=BRUSH/2
    if last:
        px,py=last
        canvas.create_line(px,py,x,y,fill="white",width=BRUSH,capstyle="round")
        pen.line((px,py,x,y),fill=255,width=BRUSH)
    canvas.create_oval(x-rad,y-rad,x+rad,y+rad,fill="white",outline="white")
    pen.ellipse((x-rad,y-rad,x+rad,y+rad),fill=255)
    last=(x,y)
    has_strokes=True
    schedule_prediction()


def lift(_):
    global last
    last=None
    schedule_prediction()

canvas.bind("<Button-1>",draw_at)
canvas.bind("<B1-Motion>",draw_at)
canvas.bind("<ButtonRelease-1>",lift)


def clear():
    global brush_image,pen,last,has_strokes,pending,preview_photo
    if pending is not None:
        root.after_cancel(pending);pending=None
    brush_image=Image.new("L",(SIZE,SIZE),0)
    pen=ImageDraw.Draw(brush_image)
    last=None; has_strokes=False
    canvas.delete("all")
    scene.itemconfig(result,text="—")
    scene.itemconfig(confidence,text="Waiting for your drawing",fill=MUTED)
    scene.itemconfig(status,text="Draw to start live recognition")
    scene.itemconfig(preview_item,image="")
    preview_photo=None
    for i in range(10): bar_targets[i]=0.0
    animate_bars()


def export_image():
    if not has_strokes:
        messagebox.showinfo("Nothing to export", "Draw a digit first.")
        return
    filename=filedialog.asksaveasfilename(title="Export your drawing",defaultextension=".png",
        filetypes=[("PNG image","*.png")],initialfile="my_digit.png")
    if filename:
        brush_image.save(filename)
        scene.itemconfig(status,text="Drawing exported as PNG")


def predict_now():
    if not has_strokes:
        messagebox.showinfo("Draw first", "Draw a digit in the canvas to make a prediction.")
    else: refresh()

action(73,623,135,36,"PREDICT",predict_now,True)
action(218,623,112,36,"CLEAR",clear)
action(340,623,152,36,"EXPORT PNG",export_image)

# Landing screen overlay; creates a brief intentional entry into the studio.
landing=tk.Frame(root,bg=BG)
landing.place(x=0,y=0,relwidth=1,relheight=1)
land_center=tk.Frame(landing,bg=BG)
land_center.place(relx=.5,rely=.47,anchor="center")
tk.Label(land_center,text="N / N",font=("Segoe UI",40,"bold"),fg=ORANGE,bg=BG).pack(pady=(0,12))
tk.Label(land_center,text="NEURALNET STUDIO",font=("Segoe UI",30,"bold"),fg=TEXT,bg=BG).pack()
tk.Label(land_center,text="HANDWRITING INTELLIGENCE  ·  BUILT FROM SCRATCH",font=("Segoe UI",11),fg=MUTED,bg=BG).pack(pady=(10,28))
tk.Button(land_center,text="ENTER STUDIO  →",command=landing.destroy,
    bg=RED,fg="white",activebackground=ORANGE,activeforeground="black",
    font=("Segoe UI",13,"bold"),padx=32,pady=15,relief="flat",cursor="hand2").pack()

root.mainloop()
