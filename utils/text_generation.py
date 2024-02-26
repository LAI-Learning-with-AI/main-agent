# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from langchain import hub
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()


def generate(input_str, system_prompt=None, chat_history_func=None, retriever=None):
    if chat_history_func and retriever:
        return generate_with_docs_and_history(input_str, system_prompt, retriever, chat_history_func)
    elif chat_history_func:
        return generate_with_history(input_str, system_prompt, chat_history_func)
    elif retriever:
        return generate_with_docs(input_str, system_prompt, retriever)
    else:
        return generate_base(input_str)


def generate_base(input_str):
    prompt = PromptTemplate.from_template("{input_str}")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=1024)
    chain = LLMChain(prompt=prompt, llm=llm)
    message = chain.invoke(input_str)
    return message["text"].strip()


def generate_with_docs(input_str, system_prompt, retriever):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=1024)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    message = retrieval_chain.invoke({"input": input_str, "context": system_prompt})
    return message['answer'].strip()


def generate_with_history(input_str, system_prompt, chat_history_func):
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                               MessagesPlaceholder(variable_name="history"),
                                               ("human", "{input}")])
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=1024)
    chain = LLMChain(prompt=prompt, llm=llm)
    with_message_history = RunnableWithMessageHistory(chain, chat_history_func, input_messages_key='input',
                                                      history_messages_key='history', output_messages_key='text')
    message = with_message_history.invoke({"input": input_str}, config={'configurable': {'session_id': 'test'}})
    return message['text'].strip()


def generate_with_docs_and_history(input_str, system_prompt, retriever, chat_history_func):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=1024)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    with_message_history = RunnableWithMessageHistory(retrieval_chain, chat_history_func, input_messages_key='input',
                                                      history_messages_key='chat_history', output_messages_key='answer')
    message = with_message_history.invoke({"input": input_str, "context": system_prompt}, config={'configurable': {'session_id': 'test'}})
    return message['answer'].strip()


def get_rating(x):
    """
    Extracts a rating from a string.
    
    Args:
    - x (str): The string to extract the rating from.
    
    Returns:
    - int: The rating extracted from the string, or None if no rating is found.
    """
    nums = [int(i) for i in re.findall(r'\d+', x)]
    if len(nums) > 0:
        return min(nums)
    else:
        return None
