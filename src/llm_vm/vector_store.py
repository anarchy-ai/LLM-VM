

class VectorStore:
    def __init__(store,query_func):
        self.store = store # some abstract database you've already loaded and indexed
        self.__query_func = query_func # query_func : (store,search string/vector?) -> List of (document,distance)
        return None 

    def query(snip):
        return self.__query_func(self.store, snip)



class DocArrayGenericStore(VectorStore):
    def __init__(store,limit,field):
        self.store = store  # underlying store
        self.limit = limit   # number of results
        self.field = field   # field in the documents that has the embedding vector 
        # these must all be 
    def query(snip): # AnyVector -> list of (doc,distance) # distance IS score right?
        if field is not None:
            retrieved_docs, scores = store.find(query=snip,search_field=self.field,limit=self.limit)
        else:
            # this only works in older versions of docarray, so no fieled 
            retrieved_docs, scores = store.find(query=snip,search_field=self.field,limit=self.limit)
        return zip(retrieved_docs,scores)            





