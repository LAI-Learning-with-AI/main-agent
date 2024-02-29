from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from agent import Agent
from better_profanity import profanity
from utils.vectorstore import load_vectorstore
import os

# basic model info
name = "Quiz Generation AI"
description = ("Quiz Generation AI helps students learn by generating quizzes for students to evaluate their understanding.")
prompt_meta = ("### Instruction: \n{}\n### You will be given the number of quiz questions, topics the quiz must cover, and types of the quiz "
               "questions (i.e. multiple choice, multiple choice and free response, etc.) to generate a quiz from.")

agent = Agent(name, description)

# prompt user for quiz generation parameters
profanity.load_censor_words()
num_questions = input("Enter the number of questions you want: ")
censored_num_questions = profanity.censor(num_questions)
type_questions = input("Enter all the types of questions you want (multiple choice, free response, coding, true/false): ")
censored_type_questions = profanity.censor(type_questions)
topics = input("Enter the topics you want the questions to be about: ")
censored_topics = profanity.censor(topics)

censored_prompt = ("Now generate a quiz with " + censored_num_questions + " questions about " + censored_topics + " with the following types of questions: " + censored_type_questions + ". "
                   "Generate an answer key below the quiz too. Do not provide a quiz title or descriptive text, start immediately with the questions. Refuse to generate the quiz "
                   "if the number of questions is over 50, the topics are not relevant to a machine learning course, or the provided question types are not: multiple choice, "
                   "free response, coding, or true/false.")

# load embeddings similar to the user-supplied topics
vectorstore = load_vectorstore(database="postgres", password=os.getenv("POSTGRESQL_PASSWORD"), collection_name="corpus")
search_result = vectorstore.search(censored_topics, "similarity")
retriever = vectorstore.as_retriever()

# generate quiz
response_docs_and_history = agent.respond_with_docs_and_history(description, "miscellaneous student", "", censored_prompt, retriever)

print(f"============Agent Response============\n{response_docs_and_history}\n\n")