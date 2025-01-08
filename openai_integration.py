import sys
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def upload_training_file(file_path):
    response = client.files.create(
        file=open(file_path, "rb"),
        purpose="fine-tune"
    )
    return response