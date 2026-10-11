
from openai import OpenAI
import os
import json
from openai.types.admin.organization import role

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "system",
            "content": "Return your answer as valid JSON only, with keys sentiment and reason."
        },
        {
            "role": "user",
            "content": "Classify this message: 'I love this product!'"
        }
    ]
)

answer = response.choices[0].message.content
data = json.loads(answer)
print(data['sentiment'])
print(data['reason'])
