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
    """
    Function to get similar documents from vector store given a list of topics.
    Useful to find resources to learn more about a topic.

    Parameters:
    - topics (list[str]): String array of topics to search for similar documents.
    - max_per_topic (int, optional): Maximum number of documents to return per topic. Defaults to 5.

    Returns:
    - list: List of dictionaries containing topic and list of similar documents.
    """
    result = []
    vs = load_vectorstore(database="postgres", password=os.getenv("POSTGRESQL_PASSWORD"),
                          collection_name="corpus")

    for t in topics:
        search = [item.metadata for item in vs.search(t, "mmr", k=max_per_topic)]
        result.append({t: search})

    return result
