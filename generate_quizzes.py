from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from better_profanity import profanity
import os
import json
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
                "choices": choices[:-1], # -2 to remove '@' at end
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



def grade_quiz(questions, temperature=0.7):
    '''Takes formatted JSON quiz and debugMode, grades all questions, and returns [total quiz score out of 1, [scores for each FRQ out of 1], [errors for each question if CODING]].'''

    num_mc = len(questions)

    # model setup
    name = "Quiz Grader"
    description = ("You are Quiz Grader. Quiz Grader helps grade free response questions.")
    agent = Agent(name, description)

    # grade all questions
    question_scores = []
    code_errors = [] # errors from running CODING questions...None if no errors or not applicable to question type
    for question in questions:

        errors = None

        # MULTIPLE_CHOICE and TRUE_FALSE: grade 0 or 1
        if question["type"] == "MULTIPLE_CHOICE" or question["type"] == "TRUE_FALSE":
            if question["answers"] == question["user_answer"]:
                question_scores.append(1)
            else:
                question_scores.append(0)
            continue

        # SHORT_ANSWER: grade [0, 1]
        if question["type"] == "SHORT_ANSWER":

            prompt1 = ("Here is a question: " + question["question"] + "\n\nHere is the optimal answer to the question:" + question["answers"] + "\n\nFrom the optimal answer, "
                    "split it up into its logical points and return them line by line, i.e. \"- Here is point 1.\n- Here is point 2.\". Only return the points with no other text.")
            answer_points = agent.respond(description, "miscellaneous student", "", prompt1, temperature=temperature)
            prompt2 = ("You will be provided with text delimited by triple quotes that is the answer to a question. Check if the following pieces of information "
                    "are mentioned in the answer:\n\n" + answer_points + "\n\nFor each piece of information, return a comma-separated \"yes\" if it is mentioned in the "
                    "answer or a comma-separated \"no\" if it is not mentioned in the answer I.e., if 3 out of 4 points are mentioned in the answer, return \"yes, yes, yes, no\". "
                    "Be lenient. Do not return any other text."
                    "\n\n\"\"\"" + question["user_answer"] + "\"\"\"")
            
            best_score = 0.0
            for i in range(3): # 3 attempts for grading, takes highest score
                response = agent.respond(description, "miscellaneous student", "", prompt2, temperature=temperature)
                score = float(response.replace(' ', '').split(',').count("yes")) / float(len(response.replace(' ', '').split(',')))
                if score > best_score:
                    best_score = score

            # print(answer_points)
            # print(response, '\n')

        # CODING: grade [0, 1]
        if question["type"] == "CODING":

            score_ratio = 0.8 # 0.8 from syntax, 0.2 from running

            # grade whether it ran
            url="http://localhost:5002/runcode"
            payload = {'code': question["user_answer"]}
            headers = {'Content-Type': 'application/json'}
            try:
                response = json.loads(requests.post(url, json=payload, headers=headers).text) # converts request to text, then parses back to JSON (instead of a Response obj)
                errors = response["errors"]
                
                # print debugging info
                print("Code Question Running - Status:")
                print("ran: ", response["ran"])
                print("errors: ", errors)
                print("status_code: ", response["status_code"], "\n")

                if response["status_code"] == 200:
                    ran_score = 1 - score_ratio
                else:
                    ran_score = 0

            except Exception as e:
                raise Exception("ERROR: caught exception in grading coding question: " + str(e))
            
            # grade general syntax
            prompt = ("You will be provided with text delimited by triple quotes that is a user's code answer to a coding question. Compare the user code to the following optimal "
                      "code answer that is delimited by double quotes:\n\n\"\"" + question["user_answer"] + "\"\"\n\nScore the user-supplied code on a continuous scale of 0.0 "
                      "to " + str(score_ratio) + " based on whether it performs the same key functionality as the optimal code. Only return the score with no other text.")
            syntax_score = float(agent.respond(description, "miscellaneous student", "", prompt, temperature=temperature))

            score = ran_score + syntax_score

        # append question score and (if applicable) errors
        question_scores.append(score)
        if errors == "" or errors == None:
            code_errors.append(None)
        else:
            code_errors.append(errors)

    # get final score
    final_score = sum(question_scores) / num_mc

    return final_score, question_scores, code_errors