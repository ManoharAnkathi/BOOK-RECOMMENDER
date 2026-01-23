📚 Semantic Book Recommender

A semantic book recommendation system that combines natural language search, vector embeddings, and emotion-based filtering to suggest books tailored to what you feel like reading.

Built with LangChain, Hugging Face sentence embeddings, Chroma vector database, and a Gradio UI.

🚀 Features

🔍 Semantic search: Describe a book in natural language (not keywords)

🧠 Vector embeddings using sentence-transformers/all-MiniLM-L6-v2

🎭 Emotion-aware ranking (Happy, Sad, Suspenseful, Angry, Surprising)

🗂️ Category filtering

🖼️ Visual gallery UI with book covers

🌐 Hugging Face–ready deployment

🧩 How It Works

Book descriptions are split into chunks and embedded using a sentence transformer

Embeddings are stored in a Chroma vector database

User queries are embedded and matched via cosine similarity

Results are:

filtered by category (optional)

re-ranked by emotional tone scores

Top recommendations are displayed in a Gradio gallery

📂 Project Structure
.
├── books_with_emotions.csv      # Book metadata + emotion scores
├── tagged_description.txt       # ISBN + tagged book descriptions
├── cover-not-found.jpg          # Fallback image
├── app.py                       # Main application
├── README.md                    # This file

📊 Dataset Requirements
books_with_emotions.csv

Must include at least the following columns:

isbn13

title

authors (semicolon-separated)

description

thumbnail

simple_categories

Emotion scores:

joy

sadness

anger

fear

surprise

tagged_description.txt

Each line should start with an ISBN followed by its description, for example:

9780143127741 A powerful story of forgiveness and redemption...


This allows the recommender to map vector search results back to book metadata.

🛠️ Installation
pip install pandas numpy gradio langchain langchain-community langchain-chroma sentence-transformers

▶️ Running the App Locally
python app.py


Then open your browser at:

http://localhost:7860

🌍 Hugging Face Deployment

This app is fully compatible with Hugging Face Spaces.

Make sure:

No /content paths are used

All data files are included in the repository

The app launches via:

dashboard.launch(server_name="0.0.0.0", server_port=7860)

🎛️ User Controls

Text query: Describe the book you want in natural language

Category filter: Optional genre filtering

Emotional tone:

Happy

Sad

Suspenseful

Angry

Surprising

🧠 Model Used

Embedding model:
sentence-transformers/all-MiniLM-L6-v2

Fast, lightweight, and well-suited for semantic search tasks.

📸 UI Preview

The interface displays recommendations as a grid gallery with:

Book cover

Title

Author(s)

Short description preview

📌 Future Improvements (Ideas)

User personalization

Hybrid keyword + semantic search

Multi-emotion blending

Click-through explanations (“Why this book?”)

📜 License

MIT License — free to use, modify, and distribute.
