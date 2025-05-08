import tkinter as tk
from tkinter import scrolledtext
from transformers import AutoTokenizer, pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
import torch
import os
import threading
from PIL import Image, ImageTk, ImageSequence

# === Device setup ===
device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# === Load model and tokenizer ===
model_name = "PY007/TinyLlama-1.1B-Chat-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
generator = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    device=device,
    torch_dtype=dtype,
    trust_remote_code=True
)

# === Load and split PDFs ===
def load_and_split_pdfs(pdf_folder):
    pdf_paths = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    
    print(f"🗂️  Total PDFs found: {len(pdf_paths)}")
    for p in pdf_paths:
        print("  -", p)

    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    print(f"\n🧩 Total chunks created: {len(chunks)}")
    if chunks:
        print("\n📘 Sample Chunk Preview:")
        print(chunks[0].page_content[:300])

    return chunks

# === Create FAISS vector store ===
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def create_faiss_index(chunks):
    return FAISS.from_documents(chunks, embedding_model)

# === Retrieve top-k relevant chunks ===
def retrieve_context(query, index, k=3):
    docs = index.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])

# === GUI setup ===
root = tk.Tk()
root.title("TinyLlama Chatbot with RAG")
root.geometry("750x550")
root.configure(bg="#2c3e50")

chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 12), bg="#ecf0f1", state=tk.DISABLED)
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
chat_window.tag_config("user", foreground="#2980b9", font=("Arial", 12, "bold"))
chat_window.tag_config("bot", foreground="#27ae60", font=("Arial", 12))
chat_window.tag_config("error", foreground="red", font=("Arial", 12))

entry_frame = tk.Frame(root, bg="#2c3e50")
entry_frame.pack(fill=tk.X, padx=10, pady=10)

entry = tk.Entry(entry_frame, font=("Arial", 12), bg="#ecf0f1")
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
entry.bind("<Return>", lambda event: start_thread())

send_button = tk.Button(entry_frame, text="Send", command=lambda: start_thread(), bg="#3498db", fg="white", font=("Arial", 12))
send_button.pack(side=tk.RIGHT)

# === Loading GIF ===
loading_label = tk.Label(root, bg="#2c3e50")
frames = [ImageTk.PhotoImage(img) for img in ImageSequence.Iterator(Image.open("loading.gif"))]
def animate_gif(index=0):
    if loading_label.winfo_ismapped():
        loading_label.configure(image=frames[index])
        root.after(100, animate_gif, (index + 1) % len(frames))

# === Generate model response with RAG ===
def get_model_response():
    user_input = entry.get().strip()
    if not user_input:
        return

    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, f"You: {user_input}\n", "user")
    chat_window.config(state=tk.DISABLED)
    chat_window.yview(tk.END)

    entry.delete(0, tk.END)
    loading_label.pack(pady=10)
    animate_gif()

    try:
        context = retrieve_context(user_input, faiss_index)
        prompt = f"""Summarize the following content or explain it in simple terms. 

{context}

### Human: {user_input}
### Assistant:"""

        result = generator(
            prompt,
            max_new_tokens=300,
            do_sample=True,
            top_k=50,
            top_p=0.7,
            repetition_penalty=1.1
        )
        response = result[0]['generated_text'].split("### Assistant:")[-1].strip()
    except Exception as e:
        response = f"Error: {e}"

    loading_label.pack_forget()
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, f"TinyLlama: {response}\n\n", "bot")
    chat_window.config(state=tk.DISABLED)
    chat_window.yview(tk.END)

def start_thread():
    threading.Thread(target=get_model_response, daemon=True).start()

# === Build or Load Vector Store ===
vectorstore_dir = "vectorstore/"
if os.path.exists(os.path.join(vectorstore_dir, "index.faiss")):
    print("📂 Loading existing FAISS index...")
    faiss_index = FAISS.load_local(vectorstore_dir, embedding_model)
else:
    print("📦 No FAISS index found. Creating a new one...")
    chunks = load_and_split_pdfs("docs")
    faiss_index = create_faiss_index(chunks)
    faiss_index.save_local(vectorstore_dir)
    print("✅ Vector store saved to disk.")

root.mainloop()
