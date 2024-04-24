# from datetime import datetime
# from utils.loaders import load_document
from better_profanity import profanity
from datetime import datetime
import os

# Fix errors when importing locally versus as submodule
if __package__ is None or __package__ == '':
    from agent import Agent
    from utils.vectorstore import load_vectorstore
else:
    from .agent import Agent
    from .utils.vectorstore import load_vectorstore

from dotenv import load_dotenv

load_dotenv()


def get_vectorstore():
    return load_vectorstore(database="postgres", password=os.getenv("POSTGRESQL_PASSWORD"),
                            collection_name="corpus")


def get_retriever():
    return get_vectorstore().as_retriever()


# debug = True

def run_chat(userid="9999", chatid="9999", message="NOMESSAGE", previous_messages=[], user_data=None, debug=False):
    print(f"Request Data: \nuserid: {userid}, \nchatid: {chatid}, \nmessage: {message}, "
          f"\nprevious_messages: {previous_messages}, \nuser_data: {user_data}\n\n")

    name = "Tutor"
    # TODO: Figure out answer in backend first, then begin the helping process
    # TODO: Consider putting description in plain-text config file so it is easier to change
    description = ("Tutor is a helpful AI assistant. He does his best to help students answer questions. He will say "
                   "\"I don't know.\" when he is unsure. He will not directly answer student questions but instead "
                   "prompt them towards the correct answer. \nIf information about the subject does not exist in the "
                   "CONTEXT, say \"I can't find a resource to help with that.\"")

    # Load Vector Store
    retriever = get_retriever()

    agent = Agent(name, description)

    user_name = "Student"
    user_description = ""  # Potentially abstract "User" into own class and update description overtime

    system_prompt = ('### Instruction: \n{}\n### Respond in a couple of sentences. Try to keep the conversation going. '
                     'Refuse to answer inappropriate questions.\n')

    profanity.load_censor_words()

    if debug:
        try:
            while True:
                if debug: print("============Main Loop============")

                # Run agent
                user_input = input("Enter your question: ")

                censored_input = profanity.censor(user_input)
                if debug: print(f"============User input============\n{censored_input}\n\n")

                response_docs_and_history = agent.respond_with_docs_and_history(system_prompt, user_name,
                                                                                user_description,
                                                                                censored_input, retriever,
                                                                                previous_messages)

                print(f"============Agent Response w/Docs&History============\n{response_docs_and_history}\n\n")

        except KeyboardInterrupt:
            print("============Exiting============")
    else:
        censored_input = profanity.censor(message)
        response_docs_and_history = agent.respond_with_docs_and_history(system_prompt, user_name, user_description,
                                                                        censored_input, retriever, previous_messages)
        return response_docs_and_history, datetime.now()


if __name__ == '__main__':
    # pMessages = ["What is Deep Learning?", "That is a complex Machine Learning Topic."]
    run_chat(debug=True)
