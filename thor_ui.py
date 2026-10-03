# thor_ui.py
import tkinter as tk
from thor_model import load_model, predict_sentiment
import math

model = load_model()


def analyze_text():
    user_text = input_box.get("1.0", tk.END).strip()
    if not user_text:
        result_label.config(text="enter text ")
        return

    predicted, probs_dict = predict_sentiment(user_text, model)

    result = f"Prediction: {predicted}\n\n"
    for cls, prob in probs_dict.items():
        result += f"{cls}: {prob*100:.2f}%\n"

    result_label.config(text=result)


root = tk.Tk()
root.title("THOR")
root.geometry("520x450")
root.configure(bg="#05070d")
root.resizable(False, False)

canvas = tk.Canvas(
    root,
    width=520,
    height=120,
    bg="#05070d",
    highlightthickness=0
)
canvas.pack()


hammer_head = canvas.create_rectangle(230, 40, 290, 70, fill="#8ab4ff", outline="")
hammer_handle = canvas.create_rectangle(258, 70, 262, 105, fill="#c7d2fe", outline="")
glow = canvas.create_oval(215, 25, 305, 115, outline="#2563eb", width=2)


angle = 0

def animate_hammer():
    global angle
    angle += 0.08

    float_y = math.sin(angle) * 4
    glow_alpha = abs(math.sin(angle))

    canvas.move(hammer_head, 0, float_y)
    canvas.move(hammer_handle, 0, float_y)
    canvas.move(glow, 0, float_y)

    canvas.itemconfig(
        glow,
        outline=f"#2563eb"
    )

    root.after(40, animate_hammer)

animate_hammer()

title = tk.Label(
    root,
    text="⚡ THOR ⚡",
    font=("Segoe UI", 20, "bold"),
    bg="#05070d",
    fg="#93c5fd"
)
title.pack(pady=(5, 0))

subtitle = tk.Label(
    root,
    text="Text-based Human Opinion Reader",
    font=("Segoe UI", 10),
    bg="#05070d",
    fg="#94a3b8"
)
subtitle.pack(pady=(0, 15))

input_box = tk.Text(
    root,
    height=4,
    width=48,
    font=("Consolas", 11),
    bg="#f8fafc",          # soft white
    fg="#020617",          # dark text
    insertbackground="#020617",
    wrap="word",
    relief="flat",
    highlightthickness=1,
    highlightbackground="#c7d2fe",
    highlightcolor="#93c5fd"
)
input_box.pack(pady=10)


analyze_btn = tk.Button(
    root,
    text="ANALYZE",
    command=analyze_text,
    font=("Segoe UI", 11, "bold"),
    bg="#2563eb",
    fg="white",
    activebackground="#1d4ed8",
    relief="flat",
    padx=18,
    pady=6
)
analyze_btn.pack(pady=12)

result_label = tk.Label(
    root,
    text="",
    font=("Consolas", 10),
    bg="#05070d",
    fg="#e5e7eb",
    justify="left"
)
result_label.pack(pady=10)

root.mainloop()
