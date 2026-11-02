
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY belum ditemukan!")

client = Groq(api_key=api_key)

print("Koneksi Groq berhasil!")

models = client.models.list()

print("\nDAFTAR MODEL YANG TERSEDIA:")

for model in models.data:
    print("-", model.id)
