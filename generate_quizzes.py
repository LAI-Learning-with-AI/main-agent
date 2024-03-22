from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from better_profanity import profanity
import os
import re

# fix errors when importing locally versus as submodule
if __package__ is None or __package__ == '':
    from agent import Agent
    from utils.vectorstore import load_vectorstore
else:
    from .agent import Agent
    from .utils.vectorstore import load_vectorstore



def _parse_quiz(quiz, numQs, topics, types):
    '''Taking a GPT-generated quiz, and the number of expected questions, parse it appropriately and return
    the quiz in parsed JSON format for HTTP API calling. Returns False instead if the quiz is not
    formatted properly for parsing.'''

    ###############
    # expected MC question format:

    # 2. In supervised learning, what is the main characteristic of the training data?
    # Topic: Supervised learning
    # Type: Multiple choice
    # A) It is labeled
    # B) It is unlabeled
    # C) It contains missing values
    # D) It is not used for training
    # Answer: A) It is labeled

    # 3. Here is another question??
    # Topic: Supervised learning
    # Type: Multiple choice
    # A) It is cat.
    # B) It is dog.
    # C) It is car.
    # D) It is bike.
    # Answer: A) It is cat.

    # 4. ...
    ###############

    body = {"questions": []}

    # traverse each question
    quiz = quiz.split("\n\n")
    for section in quiz:

        question = ""
        topic = ""
        type = ""
        choices = ""
        answer = ""

        # traverse each line
        lines = section.split("\n")
        within_answer = False # flag for handling multi-line answers (for coding questions)
        for line in lines:

            question_pattern = re.compile(r'\d+\.')
            topic_pattern = re.compile(r'Topic: ')
            type_pattern = re.compile(r'Type: ')
            answer_pattern = re.compile(r'Answer: ')
            choices_pattern = re.compile(r'[A-D]\) ') # for MC questions specifically

            # search each line for applicable patterns
            if within_answer: # for handling multi-line answers (for coding questions)
                answer += line
            if question_pattern.search(line):
                question = line.split(". ")[1]
            elif topic_pattern.search(line):
                topic = line.split(": ")[1]
            elif type_pattern.search(line):
                type = line.split(": ")[1]
            elif answer_pattern.search(line):
                within_answer = True
                answer += line.split(": ")[1]
            elif choices_pattern.search(line):
                choices += line.split(") ")[1] + ", "

        # check question-specific conditions of incorrect quiz format:
        # if question is MULTIPLE_CHOICE but does not have exactly 4 answer choices
        if (type == "MULTIPLE_CHOICE" and len(choices[:-2].split(",")) != 4):
            return False
        # if question is TRUE_FALSE but does not have exactly 2 answer choices
        if (type == "TRUE_FALSE" and len(choices[:-2].split(",")) != 2): # -2 to remove comma at end
            print(choices)
            return False
        # if question is not a valid type
        if type != "MULTIPLE_CHOICE" and type != "SHORT_ANSWER" and type != "CODING" and type != "TRUE_FALSE":
            print(type)
            return False
        # if question topic is not one of the specified topics
        topics_split = topics.split(",")
        validTopic = False
        for topic_split in topics_split:
            topic_split = topic_split[1:] if topic_split[0] == " " else topic_split # remove leading space if present
            if topic.lower() == topic_split.lower(): # .lower to be case insensitive
                validTopic = True
        if not validTopic:
            return False

        # append question to JSON list
        if type == "MULTIPLE_CHOICE" or type == "TRUE_FALSE": # these types require an extra "choices" key
            body["questions"].append({
                "type": type,
                "question": question,
                "topics": topics,
                "choices": choices[:-2], # -2 to remove comma at end
                "answer": answer
            })
        else:
            body["questions"].append({
                "type": type,
                "question": question,
                "topics": topics,
                "answer": answer
            })

    # check general conditions of incorrect quiz format:

    # if number of questions does not match number asked for in quiz generation
    if len(body["questions"]) != numQs:
        print(len(body["questions"]))
        return False
    
    return body
        


def generate_quiz(numQs, types, topics, debugMode=False):
    '''Given a numer of question, question types, question topics, and a bool debugMode, generates and
    returns a quiz using GPT. Takes 3 total attempts if the quiz is not formatted properly.
    
    If debugMode is true, returns nothing and does not attempt parsing, rather prints the raw generated
    quiz on the first attempt for debugging purposes.'''

    topics = profanity.censor(topics) # profanity check the topics

    # model setup and prompting
    name = "Quiz Generation AI"
    description = ("Quiz Generation AI helps students learn by generating quizzes for students to evaluate their understanding. "
                   "### Instructions: You will be given the number of quiz questions, topics the quiz must cover, and types of the quiz "
                   "questions (i.e. multiple choice, multiple choice and free response, etc.) to generate a quiz with.")

    agent = Agent(name, description)

    prompt = ("Make a quiz with exactly " + str(numQs) + "questions, the following question topics: " + topics + ", and "
                    "the following types of questions: " + types + ". Additional instructions:"
                    "\n\nStart immediately with question 1 and no other unnecessary text like a quiz title."
                    "\n\nNext to each question, list the question topic and type of question once, i.e.: \"5. Here is a question.\nTopic: topic1\nType: MULTIPLE_CHOICE\"."
                    "\n\nQuestion types must be one of the following: MULTIPLE_CHOICE, TRUE_FALSE, SHORT_ANSWER, CODING."
                    "\n\nMULTIPLE_CHOICE questions will list the answer choices immediately after the \"Type\" line with no whitespace, i.e.: \"A) choice1\nB) choice2\nC) choice3\nD) choice4\""
                    "\n\nTRUE_FALSE questions will list the true/false answer choices immediately after the \"Type\" line with no whitespace, similar to MULTIPLE_CHOICE. I.e.: \"Type: TRUE_FALSE\nA) True\nB) False\"."
                    "\n\nList the correct answer immediately after the answer choices (for MULTIPLE_CHOICE and TRUE_FALSE), or the question type (for SHORT_ANSWER and CODING), i.e. for MULTIPLE_CHOICE: \"D) choice4\nAnswer: "
                    "choice4\", and for all other question types, \"Type: free response\nAnswer: answer\". There should not be a blank line."
                    '\n\nFor coding questions, ensure the \"Answer: ...\" provides the full code implementation in Python and within triple apostrophes.'
                    "\n\nDo not generate a quiz if the topics are not relevant to a machine learning course.")
    
    # RAG for embeddings similar to user-supplied topics
    vectorstore = load_vectorstore(database="postgres", password=os.getenv("POSTGRESQL_PASSWORD"), collection_name="corpus")
    retriever = vectorstore.as_retriever()

    # generate quiz
    response = agent.respond_with_docs(description, "miscellaneous student", "", prompt, retriever)

    if debugMode:
        print(response)
    else:

        # parse quiz and return formatted JSON
        body = _parse_quiz(response, numQs, topics, types)

        # 2 retries if quiz is not formatted properly
        for i in range(2):
            if body == False:
                response = agent.respond_with_docs(description, "miscellaneous student", "", prompt, retriever)
                body = _parse_quiz(response, numQs, topics, types)

        return body