import os

# fix errors when importing locally versus as submodule
if __package__ is None or __package__ == '':
    from generate_quizzes import generate_quiz
    from generate_quizzes import grade_quiz
    from agent import Agent
    from utils.vectorstore import load_vectorstore
else:
    from .generate_quizzes import generate_quiz
    from .generate_quizzes import grade_quiz
    from .agent import Agent
    from .utils.vectorstore import load_vectorstore

# ======== TEST QUIZ GEN ========

# json = generate_quiz(10, 'MULTIPLE_CHOICE, SHORT_ANSWER', 'ensemble learning, dimensionality reduction', True)
# print(json)

# ======== TEST RAG VS. NO RAG RESPONSE ========

# question = 'Please explain the kernel trick in terms of Support Vector Machines and how it allows for producing a discriminant in an infinitely-high dimensional space.'
# myAgent = Agent('', '')

# response1 = myAgent.respond('', '', '', question)

# vectorstore = load_vectorstore(database="postgres", password=os.getenv("POSTGRESQL_PASSWORD"), collection_name="corpus")
# retriever = vectorstore.as_retriever()
# response2 = myAgent.respond_with_docs('', '', '', question, retriever)

# print('\n====== RESPONSE 1: ======\n' + response1)
# print('\n====== RESPONSE 2: ======\n' + response2)

# ======== TEST QUIZ GRADING ========
test_quiz = [
    {
        "questionID": "0001",
        "question": "A Lasso regularizer acts as a feature selector.",
        "type": "TRUE_FALSE",
        "answers": "True",
        "user_answer": "True",
    },
    {
        "questionID": "0002",
        "question": "Neural networks are more interpretable than linear regression.",
        "type": "TRUE_FALSE",
        "answers": "False",
        "user_answer": "True",
    },
    {
        "questionID": "0003",
        "question": "What is the difference between L1 and L2 regularization?",
        "type": "SHORT_ANSWER",
        "answers": "L1 regularization penalizes the absolute value of the coefficients, while L2 regularization penalizes the square of the coefficients. L1 regularization can perform feature selection, while L2 regularization shrinks coefficients faster, but rarely eliminates them entirely.",
        "user_answer": "L2 regularization is based on the absolute value of the coefficients, while L2 regularization is based on the square of the coefficients. An L2 regularizer penalizes the weights harsher than L1, while L1 tends to trend unnecessary feature weights to 0 and perform feature selection.",
    },
    {
        "questionID": "0004",
        "question": "In soft-margin SVM, what is the result of utilizing a very small slack variable weight?",
        "type": "SHORT_ANSWER",
        "answers": "A very small slack variable weight will result in a hard-margin SVM, which will attempt to perfectly separate the data points.",
        "user_answer": "A very small slack variable weight will allow more misclassifications, softening the margin.",
    },
    {
        "questionID": "0005",
        "question": "What is the code to print the number 5 in Python?",
        "type": "CODING",
        "answers": "print(5)",
        "user_answer": "print(5)",
    }
]

grade = grade_quiz(test_quiz) # calls grade_quiz func which in turn calls HTTP route for grading code (usually grade_quiz called within overarching Flask HTTP route)
print("(final grade, [question grades], [code errors])")
print(grade, "\n")