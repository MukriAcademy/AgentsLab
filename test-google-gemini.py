# test_gemini.py
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY", None)
print(f"API Key: {api_key[:20]}..." if api_key else "NOT SET")

if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say hello")
        print(f"✅ API Key works!")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ API Key failed: {e}")
        print("\nGet a new key from: https://aistudio.google.com/app/apikey")