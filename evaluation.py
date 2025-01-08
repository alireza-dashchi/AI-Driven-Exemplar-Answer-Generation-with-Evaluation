import pandas as pd
import json
from openai_integration import generate_exemplar_answer

def generate_answers_for_test_data(model_id):
    test_df = pd.read_json('test_data.json', lines=True)
    generated_data = []

    for _, row in test_df.iterrows():
        # Convert rubric from JSON string to dict, then to text
        rubric_dict = json.loads(row['rubric'])
        rubric_text = ", ".join(rubric_dict.get("items", []))

        prompt = f"Context: {row['task_content']}\n\nQuestion: {row['question']}\n\nRubric: {rubric_text}"
        answer = generate_exemplar_answer(model_id, prompt)

        generated_data.append({
            "question_id": row['question_id'],
            "generated_answer": answer.strip('"'),
            "actual_answer": row['answer'].strip('"')
        })

    generated_df = pd.DataFrame(generated_data)
    generated_df.to_json('generated_test_answers.json', orient='records', lines=True)
    print("Answers generated and saved to generated_test_answers.json.")