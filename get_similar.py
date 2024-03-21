import os

# fix errors when importing locally versus as submodule
if __package__ is None or __package__ == '':
    from agent import Agent
    from utils.vectorstore import load_vectorstore
else:
    from .agent import Agent
    from .utils.vectorstore import load_vectorstore


# takes in string array
def get_similar(topics: list[str], max_per_topic: int = 5) -> list:
    result = []
    vs = load_vectorstore(database="postgres", password=os.getenv("POSTGRESQL_PASSWORD"),
                          collection_name="corpus")

    for t in topics:
        result += vs.search(t, "mmr", k=max_per_topic)

    # for each item in result, add metadata["source"] to body["resources"] using list comprehension
    # only adding unique sources
    result = list({item.metadata["source"]: None for item in result}.keys())

    return result
