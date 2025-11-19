import tkinter as tk
from tkinter import scrolledtext, messagebox
import os
import datetime
import json
from llama_cpp import Llama

# ----------------------------
# Load Llama 3 model
# ----------------------------
LLAMA_MODEL_PATH = "/Users/vishwa/Downloads/Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf"

print("Loading Llama 3 model... This may take a while on CPU.")
llm = Llama(model_path=LLAMA_MODEL_PATH)
print("Model loaded successfully!")


# ----------------------------
# Chatbot Memory
# ----------------------------
MEMORY_DIR = "./chat_memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

def load_chat_history():
    path = os.path.join(MEMORY_DIR, "chat.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"history": []}

def save_chat_history(msg, response):
    session = load_chat_history()
    timestamp = datetime.datetime.now().isoformat()
    session["history"].append({"timestamp": timestamp, "user": msg, "ai": response})
    path = os.path.join(MEMORY_DIR, "chat.json")
    with open(path, "w") as f:
        json.dump(session, f, indent=2)


# ----------------------------
# Llama3 Text Generation
# ----------------------------
def llama3_generate(prompt, max_tokens=400):
    output = llm(prompt=prompt, max_tokens=max_tokens, stop=["</s>", "###"])
    response = output["choices"][0]["text"]
    save_chat_history(prompt, response)
    return response


# ----------------------------
# Chatbot App
# ----------------------------
class LlamaChatbot:
    def __init__(self, root):
        self.root = root
        self.root.title("Llama 3 Chatbot")
        self.root.geometry("900x700")

        # Title
        tk.Label(root, text="Llama3 Local Chatbot", font=("Arial", 16, "bold")).pack(pady=10)

        # Chat Display
        self.chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 12))
        self.chat_window.pack(expand=True, fill="both", padx=10, pady=10)
        self.chat_window.config(state=tk.DISABLED)

        # User input
        self.input_box = tk.Text(root, height=4, font=("Arial", 12))
        self.input_box.pack(fill="x", padx=10)
        
        # Send Button
        tk.Button(root, text="Send", font=("Arial", 12, "bold"),
                  command=self.send_message).pack(pady=10)

        # Load history
        self.load_history()

    def load_history(self):
        history = load_chat_history()["history"]
        for chat in history:
            self.show_message("You", chat["user"])
            self.show_message("Llama3", chat["ai"])

    def show_message(self, sender, message):
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_window.see(tk.END)
        self.chat_window.config(state=tk.DISABLED)

    def send_message(self):
        user_msg = self.input_box.get("1.0", tk.END).strip()
        self.input_box.delete("1.0", tk.END)

        if not user_msg:
            return

        self.show_message("You", user_msg)

        try:
            ai_response = llama3_generate(user_msg)
            self.show_message("Llama3", ai_response)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate response: {str(e)}")


# Run App
root = tk.Tk()
app = LlamaChatbot(root)
root.mainloop()
