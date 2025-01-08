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

def start_fine_tuning(training_file_id, model="gpt-4o-mini-2024-07-18"):
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=model
    )
    return response

def retrieve_fine_tuned_model(job_id):
    job = client.fine_tuning.jobs.retrieve(job_id)
    return job.fine_tuned_model