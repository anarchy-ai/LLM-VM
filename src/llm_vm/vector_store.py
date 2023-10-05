
"""
this module provides the helpers to wrap up any Vector Database that
supports a Top K embedding query interface into a RAG workflow

different but mostly similar classes are needed for enabling filtered search
which will be pretty typical

"""
# ultra generic query store interface
class VectorStore:
    def __init__(self,store,query_func):
        self.store = store # some abstract database you've already loaded and indexed
        self.__query_func = query_func # query_func : (store,search string/vector?) -> List of (document,distance)

    def query(self,snip):
        return self.__query_func(self.store, snip)


# ultra generic doc array query interface 
class DocArrayGenericStore(VectorStore):
    def __init__(self,store,limit,field):
        self.store = store  # underlying store
        self.limit = limit   # number of results
        self.field = field   # field in the documents that has the embedding vector 
        # these must all be 
    def query(self,snip): # AnyVector -> list of (doc,distance) # distance IS score right?
        if field is not None:
            retrieved_docs, scores = store.find(query=snip,search_field=self.field,limit=self.limit)
        else:
            # this only works in older versions of docarray, so no fieled 
            retrieved_docs, scores = store.find(query=snip,search_field=self.field,limit=self.limit)
        return zip(retrieved_docs,scores)            


# this is an illustrative merge and render function, there are better ones.
# also if using doc array, you can pick the scores threshold for inclusion of the result
# that renders or selects out the relevant info from the selected document.
# returns a string 
def default_merge(context,top_k,query):
    if len(top_k)>0:
        return context + "\n" + top_k[0][0] + "\n"+ query 
    else:
        return context + "\n" + query 


# this exposes a slightly different interface than onsite llm classes 
class WrappedRAG:
    def __init__(self,onsite_model,vector_store, merge_function=default_merge): 
    # takes an OnsiteLLM and a  VectorStore, and a callable that merges context + vector results into 
        self.vector_store = vector_store
        self.onsite_model = onsite_model
    def generate(self,context,query,**kwargs): # 
        top_k= vector_store.query(query)
        return onsite_model.generate(default_merge(context,top_k,query),**kwargs)



