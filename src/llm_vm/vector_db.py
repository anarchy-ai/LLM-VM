import pinecone
from scalene import profile

@profile
class PineconeDB:
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
