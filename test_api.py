from google import genai

client = genai.Client(api_key="AQ.Ab8RN6L5oQd7G9Lt0pAQGtnBhcPx7ehoPdfCecJCt4bnFL7UHg")

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Hello! Can you introduce yourself in one sentence?",
)

print("\n--- AI Response ---")
print(response.text)