from abc import ABC,abstractmethod
import pinecone
import weaviate
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv("../../.env.example")

class VectorDB(ABC):

    @abstractmethod
    def create_index(self):
        pass

    @abstractmethod
    def list_indexes(self):
        pass

    @abstractmethod
    def describe_index(self, index_name):
        pass

    @abstractmethod
    def delete_index(self, index_name):
        pass

    @abstractmethod
    def upsert(self, **kwargs):
        pass

    @abstractmethod
    def query(self, **kwargs):
        pass

import faiss

class PineconeDB(VectorDB):
    def __init__(self, api_key, pinecone_env):
        self.pinecone = pinecone
        self.pinecone.init(api_key=api_key, environment=pinecone_env)
        self.index = None
        print(self.pinecone.list_indexes())
    
    def create_index(self, **kwargs):
        if "name" not in kwargs:
            raise ValueError("Expected name as a keyword argument but found None")
        if "dimension" not in kwargs:
            kwargs["dimension"] = 1024
        if "metric" not in kwargs:
            kwargs["metric"] = "cosine"

        self.pinecone.create_index(**kwargs)
        self.index = self.pinecone.Index(kwargs["name"])

    def list_indexes(self):
        return self.pinecone.list_indexes()
    
    def describe_index(self, index_name):
        return self.pinecone.describe_index(index_name)
    
    def delete_index(self, index_name):
        self.pinecone.delete_index(index_name)
        print(f"${index_name} has been deleted")

    def upsert(self, **kwargs):
        if "vectors" not in kwargs:
            raise ValueError("Expected vectors as a keyword argument but found None")
        vec_count =  self.index.upsert(**kwargs)
        print(f"Added ${vec_count} vectors to your index")

    def query(self, **kwargs):
        return self.index.query(**kwargs)


class WeaviateDB(VectorDB):

    def __init__(self, api_key, cluster_url):
        self.client = weaviate.Client(url=cluster_url, auth_client_secret = weaviate.AuthApiKey(api_key=api_key), additional_headers={
        "X-OpenAI-Api-Key": os.environ.get("LLM_VM_OPENAI_API_KEY")
    })

    def create_index(self, class_name, class_properties=[], distance="cosine", emb_model="text2vec-openai", mod_config=None):
        assert type(class_properties) == "list"
        if mod_config is not None:
            assert type(mod_config) == "dict" 
        class_obj = {
            'class': class_name,
            'properties': [{'name': p, 'dataType': ['text']} for p in class_properties],
            "vectorizer": emb_model,
            'moduleConfig': mod_config
        }
        self.client.schema.create_class(class_obj)
        print(f"New index {class_name} created")

    def list_indexes(self):
        schemas = self.client.schema.get()
        return schemas['classes']
    
    def describe_index(self, class_name):
        return self.client.schema.get(class_name)
    
    def delete_index(self, class_name):
        self.client.schema.delete_class(class_name)
    
    def upsert(self, class_name, batch_size=50, num_workers=1, dynamic=True, dataset=[]):
        self.client.batch.configure(batch_size=batch_size, num_workers=num_workers, dynamic=dynamic)
        with self.client.batch as batch: 
            for d in dataset:
                batch.add_data_object(d, class_name)

    def query(self, prompt, class_name, top_k=3, properties=[]):
        return self.client.query.get(class_name, properties).with_near_text({"concepts": [prompt]}).with_limit(top_k).do()

    def read_object(self, class_name, obj_id):
        return self.client.data_object.get_by_id(obj_id, class_name=class_name)
    
    def read_all_objects(self, class_name):
        collection = self.client.collections.get(class_name)
        items_list = []
        for item in collection.iterator():
            items_list.append(item.properties)
        return items_list

    def add_prop(self, class_name, prop):
        assert type(prop) == "dict"
        self.client.schema.property.create(class_name, prop)     

class Faiss(VectorDB):
    def __init__(self):
        self.faiss = faiss
        self.name = None
        self.index = None
        self.dimension = 1024

    def create_index(self, name=None, dimension=None):
        if name is None:
            raise ValueError("Expected name as a keyword argument but found None")
        self.name = name
        if self.index is not None:
            raise ValueError("An index already exists")
        if dimension is not None:
            self.dimension = dimension

        self.index = self.faiss.IndexHNSWFlat(self.dimension, 32)

    def list_indexes(self):
        return [self.index] if self.index else []

    def describe_index(self):
        pass

    def delete_index(self):
        if self.index is None:
            raise ValueError("No index to delete")
        self.index = None
        print("Deleted the index")

    def upsert(self, vectors=None):
        if vectors is None:
            raise ValueError("Expected vectors as a keyword argument but found None")
        if self.index is None:
            raise ValueError("No index to add vectors to")

        self.index.add(vectors)
        print(f"Added {len(vectors)} vectors to the index")

    def query(self, vector=None, k=5):
        if vector is None:
            raise ValueError("Expected vector as a keyword argument but found None")
        if self.index is None:
            raise ValueError("No index to query")

        D, I = self.index.search(vector, k)

        if I.size > 0:
            return np.array([self.index.reconstruct(int(idx)) for idx in I[0]])
        else:
            return None

# Example of using the class
if __name__ == "__main__":
    np.random.seed(1)

    faiss_instance = Faiss()
    faiss_instance.create_index(name="test", dimension=1024)
    print(faiss_instance.list_indexes())

    data = np.random.rand(10000, 1024).astype(np.float32)
    faiss_instance.upsert(vectors=data)

    vector = np.random.rand(1, 1024).astype(np.float32)
    result = faiss_instance.query(vector=vector, k=5)
    print(result)