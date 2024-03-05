from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from better_profanity import profanity
from utils.vectorstore import load_vectorstore
import os

# fix errors when importing locally versus as submodule
if __package__ is None or __package__ == '':
    from agent import Agent
else:
    from .agent import Agent

# basic model info
name = "Quiz Generation AI"
description = ("Quiz Generation AI helps students learn by generating quizzes for students to evaluate their understanding.")
prompt_meta = ("### Instruction: \n{}\n### You will be given the number of quiz questions, topics the quiz must cover, and types of the quiz "
               "questions (i.e. multiple choice, multiple choice and free response, etc.) to generate a quiz with.")

agent = Agent(name, description)

# prompt user for quiz generation parameters
profanity.load_censor_words()
num_questions = input("Enter the number of questions you want: ")
censored_num_questions = profanity.censor(num_questions)
type_questions = input("Enter all the types of questions you want (multiple choice, free response, coding, true/false): ")
censored_type_questions = profanity.censor(type_questions)
topics = input("Enter the topics you want the questions to be about: ")
censored_topics = profanity.censor(topics)

censored_prompt = ("Make a quiz with " + censored_num_questions + " questions about " + censored_topics + " with the following question types: " + censored_type_questions + ". "
                   "Generate an answer key below the quiz. Start immediately with question 1 and no other unnecessary text. Include the topic next to each question. "
                   "Do not generate a quiz if the topics are not relevant to a machine learning course.")

# load embeddings similar to the user-supplied topics
vectorstore = load_vectorstore(database="postgres", password=os.getenv("POSTGRESQL_PASSWORD"), collection_name="corpus")
search_result = vectorstore.search(censored_topics, "similarity")
retriever = vectorstore.as_retriever()

# generate quiz
response_docs = agent.respond_with_docs(description, "miscellaneous student", "", censored_prompt, retriever)

print(f"============Agent Response============\n{response_docs}\n\n")