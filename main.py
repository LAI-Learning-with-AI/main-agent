from agent import Agent
from datetime import datetime
from utils.loaders import load_document
from better_profanity import profanity

from dotenv import load_dotenv
load_dotenv()

debug = True
def main():
    print("============Starting============")

    name = "Tutor"
    # TODO: Figure out answer in backend first, then begin the helping process
    # TODO: Consider putting description in plain-text config file so it is easier to change
    description = ("Tutor is a helpful AI assistant. He does his best to help students answer questions. His subject "
                   "focus is in AI and Machine Learning. He will say \"I don't know.\" when he is unsure. He will not "
                   "directly answer student questions but instead prompt them towards the correct answer. He refuses to"
                   "answer questions not about Artificial Intelligence, Machine Learning, Computer Science, "
                   "or something in the field. When a subject he doesn't know about comes"
                   "up, he will say \"I can't help with that.\".")

    # Load documents
    docs = []
    docs += load_document("./ml_materials/videos/3blue1brown_1_what_is_a_nn.txt")

    # TODO: Setup local vector store

    agent = Agent(name, description)

    # user_name = input("Enter name: ")
    user_name = "Student"
    user_description = ""  # Potentially abstract "User" into own class and update description overtime

    prompt_meta = ('### Instruction: \n{}\n### Respond in a couple of sentences. Try to keep the conversation going. '
                   'Refuse to answer inappropriate questions.\n'
                   'Response:')

    profanity.load_censor_words()

    try:
        while True:
            if debug: print("============Main Loop============")

            # Run agent
            user_input = input("Enter your question: ")
            censored_input = profanity.censor(user_input)
            if debug: print(f"============User input============\n{censored_input}\n\n")
            response = agent.respond(prompt_meta, user_name, user_description, censored_input)
            print(f"============Agent response============\n{response}\n\n")

            # Update memories
            #agent.add_memory(user_name, censored_input)
            #agent.add_memory(agent.name, response)
            #if debug: print(f"============{agent.name} remembers============\n{agent.memories[-2]}\n{agent.memories[-1]}\n\n")

            # TODO: Refactor reflection system to be better suited to the learning process (reflect on what the
            #  student struggles with) Reflect on memories
            # if agent.should_reflect():
            #     agent.reflect_on_memories(prompt_meta)
            #     if debug: print(f"============{agent.name} reflections============\n"
            #                     f"{' '.join([str(memory) for memory in agent.memories[-3:]])}\n\n")

    except KeyboardInterrupt:
        print("============Exiting============")


if __name__ == '__main__':
    main()
