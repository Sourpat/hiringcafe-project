from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
k = (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or "").strip().strip('"').strip("'")

print("present", bool(k), "prefix", k[:7], "len", len(k))
client = OpenAI(api_key=k)
print("calling models.list...")
print(client.models.list().data[0].id)
