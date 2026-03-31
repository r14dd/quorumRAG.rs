import wikipediaapi
import os

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='QuorumRAG/1.0'
)

articles = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Neural network",
    "Natural language processing",
    "Computer vision",
    "Transformer (machine learning model)",
    "Large language model",
    "Reinforcement learning",
    "Convolutional neural network",
    "Recurrent neural network",
    "Generative adversarial network",
    "BERT (language model)",
    "GPT (language model)",
    "Backpropagation",
    "Support vector machine",
    "Random forest",
    "Gradient boosting",
    "K-means clustering",
    "Principal component analysis",
    "Computer science",
    "Algorithm",
    "Data structure",
    "Operating system",
    "Computer network",
    "Internet",
    "World Wide Web",
    "Database",
    "Cryptography",
    "Quantum computing",
]

os.makedirs("data/corpus", exist_ok=True)

for title in articles:
    page = wiki.page(title)
    if page.exists():
        filename = title.replace(" ", "_").replace("/", "_") + ".txt"
        with open(f"data/corpus/{filename}", "w") as f:
            f.write(page.text)
        print(f"Downloaded: {title}")
    else:
        print(f"Not found: {title}")

print(f"\nDone. {len(os.listdir('data/corpus'))} articles saved.")
