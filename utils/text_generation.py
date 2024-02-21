# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()


def generate(input_str):
    prompt = PromptTemplate.from_template("{input_str}")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=1024)
    chain = LLMChain(prompt=prompt, llm=llm)
    message = chain.invoke(input_str)
    return message["text"].strip()


def generate_with_docs(input_str, retriever):
    # prompt = PromptTemplate.from_template("{context} {input_str}")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=1024)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    message = retrieval_chain.invoke({"input": input_str})
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
