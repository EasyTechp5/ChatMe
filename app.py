from flask import Flask, render_template, request, session
from utils import load_and_store_pdf, query_bot
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Optional: confirm your key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("Using Google API Key:", GOOGLE_API_KEY)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session

# Load knowledge base once at startup
pdf_path = "data/FaQ_for_EasyTech.pdf"
qa_chain = load_and_store_pdf(pdf_path)

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        user_query = request.form["query"]

        # Get bot response
        bot_response = query_bot(qa_chain, user_query)

        # Save to history
        session["chat_history"].append(("You", user_query))
        session["chat_history"].append(("Bot", bot_response))
        session.modified = True  # Mark session updated

    return render_template("index.html", chat_history=session["chat_history"])
    
if __name__ == "__main__":
    app.run(debug=True)
