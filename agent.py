from langchain_community.chat_message_histories import ChatMessageHistory
from datetime import datetime

if __package__ is None or __package__ == '':
    from utils.text_generation import generate, generate_with_docs
else:
    from .utils.text_generation import generate, generate_with_docs

debug = True


class Agent:
    def __init__(self, name, description, memory_reflection_threshold=20):
        self.name = name
        self.description = description

    def __repr__(self):
        return f"Agent({self.name}, {self.description})"

    def respond_with_docs(self, prompt_meta, user_name, user_description, user_input, retriever, temperature=0.7):
        now = datetime.now()

        prompt = f"You are {self.name}. {self.description} It is currently {now}. You are interacting with {user_name}. "
        response = generate(user_input, prompt_meta.format(prompt), retriever=retriever, temperature=temperature)

        if debug: print(f"============Agent Prompt============\n{prompt}\n\n")

        return response

    def respond_with_docs_and_history(self, system_prompt, user_name, user_description, user_input, retriever, messages, temperature=0.7):
        prompt = f"You are {self.name}. {self.description} You are interacting with {user_name}. "

        # Build chat history
        chat_history = ChatMessageHistory(messages=[])  # remove messages=[] if causing issues
        for i in range(len(messages)):
            if i % 2 == 0:
                chat_history.add_user_message(messages[i])
            else:
                chat_history.add_ai_message(messages[i])

        def get_chat_history(session_id: str = None):
            return chat_history

        response = generate(user_input, system_prompt.format(prompt), get_chat_history, retriever, temperature=temperature)

        if debug: print(f"============Chat History============\n{get_chat_history()}\n\n")
        if debug: print(f"============Agent Prompt============\n{prompt}\n\n")

        return response
