from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import OpenAIEmbeddings


def load_vectorstore_helper(connection_string, collection_name="embeddings", embeddings_function=OpenAIEmbeddings()):
    return PGVector(
        collection_name=collection_name,
        connection_string=connection_string,
        embedding_function=embeddings_function,
    )


def load_vectorstore(host="localhost", port=5432, driver="psycopg2", user="postgres", password="postgres",
                     database="postgres", collection_name="embeddings"):
    connection_string = PGVector.connection_string_from_db_params(
        driver=driver,
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
    )
    return load_vectorstore_helper(connection_string, collection_name)
