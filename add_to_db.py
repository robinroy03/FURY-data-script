""""
This script adds all the given content to the vector db
"""

import os
import sys

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from extractor import traverse_directory_tree

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
index = pc.Index(os.environ.get("PINECONE_DB_INDEX"))


def main(path: str = "fury"):
    data = traverse_directory_tree(path)

    for i in tqdm(range(len(data))):
        content = data[i]
        if content['type'] == "rst":
            for j in range(len(content['content'])):
                model_encode_data = {"type": "rst"} | {"path": content['path']} | {"content": content['content'][j]}
                index.upsert(
                    vectors=[{
                        "id": str(i),
                        "values": model.encode(str(model_encode_data)),
                        "metadata": {"data": str(content)}
                    }]
                )
        elif content['type'] == 'documentation_examples':
            for i in range(len(data)):
                content = data[i]
                for j in range(len(content['content'])):
                    model_encode_data = {"type": content['type']} | {"path": content['path']} | {"content": content['content'][j]}
                    index.upsert(
                        vectors=[{
                            "id": str(i),
                            "values": model.encode(str(model_encode_data)),
                            "metadata": {"data": str(content)}
                        }]
                    )
        else:
            index.upsert(
                vectors=[{
                    "id": str(i),
                    "values": model.encode(str(content)),
                    "metadata": {"data": str(content)}
                }]
            )

if __name__ == "__main__":
    main(sys.argv[1])

    # example: python add_to_db.py fury
