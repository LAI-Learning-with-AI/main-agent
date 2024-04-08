from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from better_profanity import profanity
import os
import re
import requests

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
    split_quiz = quiz.split("------DIVIDER------\n")
    for section in split_quiz:

        question = ""
        topic = ""
        type = ""
        choices = ""
        answer = ""

        # skip blank question sections (i.e. if divider is at end of quiz)
        if section == "" or section == "\n": continue

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
                choices += line.split(") ")[1] + "@"

        # check question-specific conditions of incorrect quiz format:
        # if question is MULTIPLE_CHOICE but does not have exactly 4 answer choices
        if (type == "MULTIPLE_CHOICE" and len(choices[:-2].split("@")) != 4):
            print('ERROR: mc question does not have exactly 4 choices\nCHOICES: ' + choices + '\nQUIZ:\n' + quiz)
            return False
        # if question is TRUE_FALSE but does not have exactly 2 answer choices
        if (type == "TRUE_FALSE" and len(choices[:-2].split("@")) != 2): # -2 to remove comma at end
            print('ERROR: tf question does not have exactly 2 choices\nCHOICES: ' + choices + '\nQUIZ:\n' + quiz)
            return False
        # if question is not a valid type
        if type != "MULTIPLE_CHOICE" and type != "SHORT_ANSWER" and type != "CODING" and type != "TRUE_FALSE":
            print('ERROR: question type is not valid\nTYPE: ' + type + '\nQUIZ:\n' + quiz)
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
                "topics": topic,
                "choices": choices[:-2], # -2 to remove '@' at end
                "answer": answer
            })
        else:
            body["questions"].append({
                "type": type,
                "question": question,
                "topics": topic,
                "choices": None,
                "answer": answer
            })

    # check general conditions of incorrect quiz format:

    # if number of questions does not match number asked for in quiz generation
    if len(body["questions"]) != numQs:
        print('ERROR: generated num of questions does not match expected num\nEXPECTED: ' + str(numQs) + '\nGENERATED: ' + str(len(body['questions'])) + '\nQUIZ:\n' + quiz)
        return False
    
    return body
        


def generate_quiz(numQs, types, topics, seeRawQuiz=False):
    '''Given a numer of question, question types, question topics, and a bool debugMode, generates and
    returns a quiz using GPT. Takes 3 total attempts if the quiz is not formatted properly, and will
    ultimately return False if a proper quiz is not generated.
    
    If seeRawQuiz is true, prints the raw generated quiz independently, before trying parsing.'''

    topics = profanity.censor(topics) # profanity check the topics

    # model setup and prompting
    name = "Quiz Generation AI"
    description = ("You are Quiz Generation AI. Quiz Generation AI is given a number of quiz questions, quiz topics, and quiz question "
                   "types from which to generate a quiz.")
    
    agent = Agent(name, description)

    prompt = ("Make a quiz with exactly " + str(numQs) + "questions on the following question topics: " + topics + ", and only "
              "using the following types of questions: " + types + ". Additional instructions: "
              "\n\nStart immediately with the first question and no other unnecessary text like a quiz title, i.e. \"1. How does regularization work?.\""
              "\n\nOn the line immediately after the question, list the question topic, i.e. \"Topic: topic1\"."
              "\n\nOn the line immediately after the topic, list the question type, i.e. \"Type: MULTIPLE_CHOICE\". The type should only be one of the aforementioned types requested."
              "\n\nFor MULTIPLE_CHOICE questions, list exactly 4 answer choices immediately following the topic, i.e. \"A) choice1\nB) choice2\nC) choice3\nD) choice4\"."
              "\n\nFor TRUE_FALSE questions, also list the 2 true/false options immediately following the topic, i.e. \"A) True\nB) False\"."
              "\n\nOn the immediate next line, list the answer to the question, i.e.: \"Answer: A) True\" or \"Answer: choice1\". For CODING question answers, list the full code implementation in Python within triple apostrophes."
              "\n\nEntire questions must be separated by a line with the text \"------DIVIDER------\" and nothing else."
              "\n\nDo not generate the quiz if the topics are highly irrelevant to a machine learning course, i.e. \"Ponies\".")

                # "\n\nNext to each question, list the question topic and type of question once, i.e.: \"5. Here is a question.\nTopic: topic1\nType: MULTIPLE_CHOICE\"."
                # "\n\nMULTIPLE_CHOICE questions will list the answer choices immediately after the \"Type\" line with no whitespace, i.e.: \"A) choice1\nB) choice2\nC) choice3\nD) choice4\""
                # "\n\nTRUE_FALSE questions will list the true/false answer choices immediately after the \"Type\" line with no whitespace, similar to MULTIPLE_CHOICE. I.e.: \"Type: TRUE_FALSE\nA) True\nB) False\"."
                # "\n\nFor CODING questions, ensure the \"Answer: ...\" provides the full code implementation in Python and within triple apostrophes."
                # "\n\nSHORT_ANSWER questions should not pertain to any code or code implementations."
                # "\n\nFor all questions, list the answer on the last relevant line for the question, i.e.: \"...\nAnswer: \". There should not be a blank line before the answer."
                # "\n\nDo not generate a quiz if the topics are not relevant to a machine learning course.")
    
    # RAG for embeddings similar to user-supplied topics
    vectorstore = load_vectorstore(database="postgres", password=os.getenv("POSTGRESQL_PASSWORD"), collection_name="corpus")
    retriever = vectorstore.as_retriever()

    # generate quiz
    print('\n========== GENERATION 1 ==========\n')
    response = agent.respond_with_docs(description, "miscellaneous student", "", prompt, retriever)

    if seeRawQuiz:
        print(response)

    # parse quiz and return formatted JSON
    try: # catch hard errors (logic errors handled in _parse_quiz)
        body = _parse_quiz(response, numQs, topics, types)
    except Exception as e:
        print("ERROR: caught exception in parsing quiz: " + str(e))
        body = False

    # 2 retries if quiz is not formatted properly
    for i in range(2):
        if body == False:
            print('\n========== GENERATION ' + str(i+2) + ' ==========\n')
            response = agent.respond_with_docs(description, "miscellaneous student", "", prompt, retriever)
            body = _parse_quiz(response, numQs, topics, types)

    return body



def grade_quiz(questions):
    '''Takes formatted JSON quiz and debugMode, grades all questions, and returns [total quiz score out of 1, [scores for each FRQ out of 1]].'''

    num_mc = len(questions)

    # model setup
    name = "Quiz Grader"
    description = ("You are Quiz Grader. Quiz Grader helps grade free response questions.")
    agent = Agent(name, description)

    # grade all questions
    question_scores = []
    for question in questions:

        # MULTIPLE_CHOICE and TRUE_FALSE: grade 0 or 1
        if question["type"] == "MULTIPLE_CHOICE" or question["type"] == "TRUE_FALSE":
            if question["answer"] == question["user_answer"]:
                question_scores.append(1)
            else:
                question_scores.append(0)
            continue

        # SHORT_ANSWER: grade [0, 1]
        if question["type"] == "SHORT_ANSWER":
            # TODO: option 1
            # prompt1 = ("Here is a question: " + question["question"] + "\n\nHere is the optimal answer to the question:" + question["answers"] + "\n\nFrom the optimal answer, "
            #         "split it up into its logical points and return them line by line, i.e. \"- Here is point 1.\n- Here is point 2.\". Only return the points with no other text.")
            # answer_points = agent.respond(description, "miscellaneous student", "", prompt1)
            # prompt2 = ("You will be provided with text delimited by triple quotes that is supposed to be the answer to a question. Check if the following pieces of information "
            #         "are directly contained in the answer:\n\n" + answer_points + "\n\nFor each piece of information, consider if someone reading the text who doesn't know the topic "
            #         "could directly infer the point. Count \"yes\" if the answer is yes, otherwise count \"no\". Only return the ratio of \"yes\" to the total number of points, and "
            #         "no other text, i.e. \"0.75\".\n\n\"\"\"" + question["user_answer"] + "\"\"\"")
            # score = float(agent.respond(description, "miscellaneous student", "", prompt2))

            # TODO: option 2
            prompt = ("Here is a question: " + question["question"] + "\n\nHere is the optimal answer to the question:" + question["answers"] + "\n\nHere is a user-supplied"
                    "answer to the question: " + question["user_answer"] + "\n\nScore the user-supplied answer on a continuous scale of 0.0 to 1.0 based on how correct it is. "
                    "A user-supplied answer that does not mention the main points of the optimal answer should receive a score of 0.0. A user-supplied answer that mentions all the"
                    "main points of the optimal answer should receive a score of 1.0. Only return the score with no other text.")
            score = float(agent.respond(description, "miscellaneous student", "", prompt))

        # CODING: grade [0, 1]
        if question["type"] == "CODING":

            # grade whether it ran
            url="http://localhost:5002/runcode"
            payload = {'code': question["user_answer"]}
            headers = {'Content-Type': 'application/json'}
            try:
                response = requests.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    ran_score = 0.5
                else:
                    ran_score = 0
            except Exception as e:
                raise Exception("ERROR: caught exception in grading coding question: " + str(e))
            
            # grade general syntax
            # TODO: option 2
            prompt = ("Here is a question: " + question["question"] + "\n\nHere is the optimal code to answer the question:" + question["answers"] + "\n\nHere is the user-supplied "
            "code to answer the question: " + question["user_answer"] + "\n\nScore the user-supplied code on a continuous scale of 0.0 to 0.5 based on how correct it is. Correct code "
            "will perform the same basic function as the optimal code, but may not be exactly the same. Only return the score with no other text.")
            syntax_score = float(agent.respond(description, "miscellaneous student", "", prompt))

            score = ran_score + syntax_score

        question_scores.append(score)

    # get final score
    final_score = sum(question_scores) / num_mc

    return final_score, question_scores