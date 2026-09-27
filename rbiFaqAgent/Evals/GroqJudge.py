from deepeval.models import DeepEvalBaseLLM
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os


class GroqJudge(DeepEvalBaseLLM):
    load_dotenv()

    def __init__(self, *args, **kwargs):
        self.client = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.1,
            api_key=os.environ.get("GROQ_API_KEY")
        )
        super().__init__(*args, **kwargs)

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.client.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return "Groq Judge"