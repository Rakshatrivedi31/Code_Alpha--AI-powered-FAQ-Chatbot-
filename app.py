from flask import Flask, render_template, request, jsonify
import math
import string

app = Flask(__name__)

# ─────────────────────────────────────────────────────────
# FAQ DATABASE — University Support
# ─────────────────────────────────────────────────────────
FAQS = [
    {"q": "How do I apply for admission?",
     "a": "Applications are open on our portal at admissions.greenfield.edu. Upload your Class 10 & 12 marksheets and pay the ₹500 registration fee. Shortlisted students will be called for counselling. 📋"},

    {"q": "What is the last date for admission?",
     "a": "The admission portal closes on July 31st each year. We recommend applying early as seats fill up quickly!"},

    {"q": "What are the eligibility criteria for admission?",
     "a": "For UG programs, minimum 60% in Class 12. For PG, minimum 55% in graduation. Engineering programs require a valid JEE / MHT-CET score."},

    {"q": "What are the tuition fees?",
     "a": "Fee structure:\n• B.Tech — ₹1,20,000/year\n• BBA/BCA — ₹80,000/year\n• MBA — ₹1,50,000/year\n• B.Sc — ₹60,000/year\nFees can be paid semester-wise. 💰"},

    {"q": "Are scholarships available?",
     "a": "Yes! Merit Scholarship (top 10% get 50% fee waiver), need-based grants for EWS students, and sports quota seats. Apply within 30 days of admission. 🎁"},

    {"q": "What hostel facilities are available?",
     "a": "Separate hostels for boys and girls with AC & non-AC rooms. Includes 24/7 Wi-Fi, dining mess, laundry, gym, and CCTV security. Fee: ₹80,000–₹1,10,000/year including meals. 🏠"},

    {"q": "When are the exams scheduled?",
     "a": "Semester exams: November–December (Odd Sem) and April–May (Even Sem). Timetables are posted on the ERP portal 3 weeks before exams. 📅"},

    {"q": "How do I check my exam results?",
     "a": "Results are published on erp.greenfield.edu within 30 days of exams. You'll also get an SMS & email on your registered number. 📊"},

    {"q": "What if I fail in an exam?",
     "a": "Students can appear in the supplementary/backlog exam held every June–July. Maximum 3 attempts allowed per subject. Register via the exam office."},

    {"q": "How is the placement process?",
     "a": "Placement Cell starts from 3rd year. Top recruiters: TCS, Infosys, Amazon, Wipro, Deloitte.\n📈 Average Package: ₹5.4 LPA\n🏆 Highest Package: ₹24 LPA\nRegister on the placement portal in 3rd year. 💼"},

    {"q": "How do I get my degree certificate?",
     "a": "Degree certificates are issued at the annual Convocation Ceremony. Apply online 2 months before convocation. Marksheets dispatched within 60 days of result declaration. 📜"},

    {"q": "What library facilities are available?",
     "a": "Central library has 50,000+ books, e-journals, NPTEL access, and 200 reading seats. Open Mon–Sat, 8 AM–10 PM. Borrow up to 4 books for 15 days with your ID card. 📚"},

    {"q": "What is the attendance requirement?",
     "a": "Minimum 75% attendance is mandatory to appear in semester exams. Students below 75% will be debarred. Medical exemptions require proper documentation."},

    {"q": "How do I get a bonafide or transfer certificate?",
     "a": "Visit the Admin Office (Room 101, Admin Block) with your student ID and a written application. Processing time: 3–5 working days. TC requires a No-Dues clearance. 📄"},

    {"q": "What clubs and extracurriculars are there?",
     "a": "25+ active clubs including Coding Club, Drama Society, Music Band, NSS, Debate Team, and Entrepreneurship Cell. Join during the Club Fair in August! 🎭"},

    {"q": "How do I access Wi-Fi on campus?",
     "a": "Connect to 'GFU-Campus' Wi-Fi with your student ID and ERP password. 100 Mbps available 24/7 in hostels, 8 AM–10 PM in academic buildings. 📶"},

    {"q": "What medical facilities are available?",
     "a": "On-campus Health Center open Mon–Sat, 9 AM–6 PM. First aid available 24/7 at hostels. Tie-up with City General Hospital for emergencies. Free OPD for students. 🏥"},

    {"q": "Is there a sports facility?",
     "a": "Cricket ground, football field, basketball & badminton courts, and a swimming pool. Sports complex open 6 AM–8 PM daily. Inter-college tournaments every semester. ⚽"},

    {"q": "How do I apply for leave?",
     "a": "Submit leave application on ERP portal at least 2 days in advance. Medical leaves need a doctor's certificate. Leaves beyond 3 days require HOD approval. 🗓️"},

    {"q": "Fee concession for economically weaker students?",
     "a": "Students with family income below ₹3L/year can get up to 70% fee concession. Submit income certificate, Aadhaar, and marksheets to the Scholarship Office within 45 days of admission. 🤝"},
]

# ─────────────────────────────────────────────────────────
# NLP PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────
STOPWORDS = set([
    'i','me','my','we','our','you','your','he','she','it','they','them',
    'what','which','who','this','that','am','is','are','was','were','be',
    'been','have','has','had','do','does','did','a','an','the','and','but',
    'if','or','of','at','by','for','with','to','in','on','so','not','no',
    'how','can','will','would','could','should','please','much','just',
    'also','as','there','any','about','tell','want','know','get','give',
    'when','where','are','us'
])

def stem(word):
    """Simple Porter-style stemmer"""
    rules = [
        ('ational','ate'), ('tional','tion'), ('nesses',''), ('ness',''),
        ('ments',''), ('ment',''), ('ings',''), ('ing',''), ('ation','ate'),
        ('ies','y'), ('ied','y'), ('ers',''), ('er',''), ('ly',''),
        ('ed',''), ('s','')
    ]
    if len(word) <= 3:
        return word
    for suffix, replacement in rules:
        if word.endswith(suffix):
            stemmed = word[:-len(suffix)] + replacement
            if len(stemmed) > 2:
                return stemmed
    return word

def preprocess(text):
    """
    Full NLP Pipeline:
    1. Lowercase
    2. Remove punctuation
    3. Tokenize
    4. Remove stopwords
    5. Apply stemming
    """
    text = text.lower()
    text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
    tokens = text.split()
    tokens = [stem(t) for t in tokens if len(t) > 1 and t not in STOPWORDS]
    return tokens

# ─────────────────────────────────────────────────────────
# TF-IDF + COSINE SIMILARITY
# ─────────────────────────────────────────────────────────
def compute_idf(corpus):
    idf = {}
    N = len(corpus)
    for doc in corpus:
        for term in set(doc):
            idf[term] = idf.get(term, 0) + 1
    for term in idf:
        idf[term] = math.log(N / (1 + idf[term])) + 1
    return idf

def tfidf_vector(tokens, idf):
    tf = {}
    n = len(tokens) or 1
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return {t: (tf[t] / n) * idf.get(t, 0.5) for t in tf}

def cosine_similarity(v1, v2):
    keys = set(v1) | set(v2)
    dot  = sum(v1.get(k, 0) * v2.get(k, 0) for k in keys)
    mag1 = math.sqrt(sum(x**2 for x in v1.values()))
    mag2 = math.sqrt(sum(x**2 for x in v2.values()))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0

# Build index at startup
corpus   = [preprocess(f["q"]) for f in FAQS]
IDF      = compute_idf(corpus)
FAQ_VECS = [tfidf_vector(tokens, IDF) for tokens in corpus]

def get_best_answer(user_query, threshold=0.12):
    tokens = preprocess(user_query)
    if not tokens:
        return None
    query_vec = tfidf_vector(tokens, IDF)
    scores    = [cosine_similarity(query_vec, fv) for fv in FAQ_VECS]
    best_idx  = scores.index(max(scores))
    if scores[best_idx] >= threshold:
        return FAQS[best_idx]["a"]
    return None

# ─────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data      = request.get_json()
    user_msg  = data.get("message", "").strip()

    if not user_msg:
        return jsonify({"answer": "Please type something! 😊"})

    answer = get_best_answer(user_msg)

    if answer:
        return jsonify({"answer": answer})
    else:
        return jsonify({"answer": "I'm sorry, I couldn't find an answer to that. Please call us at 1800-XXX-XXXX or email support@greenfield.edu and our team will help you shortly! 😊"})


if __name__ == "__main__":
    print("\n✅ Greenfield University FAQ Chatbot")
    print("👉 Open in browser: http://localhost:5000\n")
    app.run(debug=True, port=5000)