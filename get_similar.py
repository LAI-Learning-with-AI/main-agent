import os

# fix errors when importing locally versus as submodule
if __package__ is None or __package__ == '':
    from agent import Agent
    from utils.vectorstore import load_vectorstore
else:
    from .agent import Agent
    from .utils.vectorstore import load_vectorstore


# takes in string array
def get_similar(topics: list[str], max_per_topic: int = 5) -> dict[str, list[str]]:
    """
    Given a list of topics, this function retrieves the most similar documents from a PostgreSQL vector store.

    Parameters:
        topics (list[str]): A list of topics for which to retrieve similar documents.
        max_per_topic (int, optional): The maximum number of similar documents to retrieve per topic. Defaults to 5.

    Returns:
        dict[str, list[str]]: A dictionary mapping each topic to a list of the most similar documents.
    """

    result = {}
    vs = load_vectorstore(database="postgres", password=os.getenv("POSTGRESQL_PASSWORD"),
                          collection_name="corpus")

    for t in topics:
        # result[t] = vs.search(t, "mmr", k=max_per_topic)
        result[t] = [item.metadata["source"] for item in vs.search(t, "mmr", k=max_per_topic)]

    return result
