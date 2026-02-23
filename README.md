# 🎓 University FAQ Chatbot — CodeAlpha Internship

A professional FAQ Chatbot for a University Support System built during my internship at **CodeAlpha**.

## 🚀 Features
- Full NLP Pipeline: Tokenization → Stopword Removal → Stemming
- TF-IDF Vectorization + Cosine Similarity for intelligent query matching
- 20+ University FAQs (Admissions, Fees, Exams, Hostel, Placements & more)
- Clean professional UI — similar to real-world support portals
- Built with Flask (no external NLP libraries needed!)

## 🛠️ Tech Stack
`Python` `Flask` `HTML` `CSS` `JavaScript`

## ▶️ How to Run

```bash
# 1. Install Flask
pip install flask

# 2. Run the app
python app.py

# 3. Open in browser
http://localhost:5000
```


## 🧠 How It Works
```
User Input
    ↓ Lowercase + Clean
    ↓ Tokenize
    ↓ Remove Stopwords
    ↓ Porter Stemming
    ↓ TF-IDF Vectorization
    ↓ Cosine Similarity
    ↓ Best Match Answer
```

---
Made with ❤️ during CodeAlpha Internship
