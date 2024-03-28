import os

# fix errors when importing locally versus as submodule
if __package__ is None or __package__ == '':
    from generate_quizzes import generate_quiz
    from agent import Agent
    from utils.vectorstore import load_vectorstore
else:
    from .generate_quizzes import generate_quiz
    from .agent import Agent
    from .utils.vectorstore import load_vectorstore

# ======== TEST QUIZ GEN ========

# json = generate_quiz(10, 'MULTIPLE_CHOICE, SHORT_ANSWER', 'ensemble learning, dimensionality reduction', True)
# print(json)

# ======== TEST RAG VS. NO RAG RESPONSE ========

question = 'Please explain the kernel trick in terms of Support Vector Machines and how it allows for producing a discriminant in an infinitely-high dimensional space.'
myAgent = Agent('', '')

response1 = myAgent.respond('', '', '', question)

vectorstore = load_vectorstore(database="postgres", password=os.getenv("POSTGRESQL_PASSWORD"), collection_name="corpus")
retriever = vectorstore.as_retriever()
response2 = myAgent.respond_with_docs('', '', '', question, retriever)

print('\n====== RESPONSE 1: ======\n' + response1)
print('\n====== RESPONSE 2: ======\n' + response2)

# ======== TEST QUIZ GRADING ========
# TODO: