from google import genai

client = genai.Client(api_key="ENTER YOUR API KEY")

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Hello! Can you introduce yourself in one sentence?",
)

print("\n--- AI Response ---")
print(response.text)
