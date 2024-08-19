from openai import OpenAI


class OpenAIClient:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=max_new_tokens,
            logprobs=False,
        )

        return response.choices[0].message.content
