from langchain_community.chat_models import ChatOllama

def load_llm(api_key=None):
    return ChatOllama(
        model="mistral",
        temperature=0.2
    )

