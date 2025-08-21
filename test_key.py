from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("Using Google API Key:", GOOGLE_API_KEY)
