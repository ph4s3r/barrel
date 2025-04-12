import os
import pickle
import pprint
from collections import Counter

from pinecone import Pinecone

from credentials.secrets import secrets


class PineConeClient:

    def __init__(self) -> None:
        self.max_batch_size = 100
        self.index_name = "voyage1024"
        self.index_host = f"https://{self.index_name}-226a147.svc.aped-4627-b74a.pinecone.io"
        self.namespace = "vnets1024"
        self.cached_vectors = {}

        # this value should be coming from the local cache
        self.cached_vectors_count = None
        self.local_cache = "cache/vectors.dict"
        self.cache_refreshed_already = True

        # this data should be coming from the api
        self.ns_vectorcount = None
        self.pc = Pinecone(api_key=secrets.vector_db_api_key)
        self.index = self.pc.Index(name=self.index_name, host=self.index_host)
        self.stats = None
        print("successfully connected to PineCone!")
        self._refresh_index_stats()

        self.read_units_used = 0

        if not self._cache_synced():
            if not self.cache_refreshed_already:
                self.refresh_cache()
            else:
                print("Something went wrong with the cache refresh, please reach out to your admin to refresh manually.")
        else:
            print("Cache is synced")

    def _refresh_index_stats(self) -> None:
        try:
            self.stats = self.index.describe_index_stats()
            print("successfully refreshed index stats:")
            pprint.pprint(self.stats)

            self.ns_vectorcount = self.stats["namespaces"][self.namespace]["vector_count"]
        except Exception as err:
            os._exit(f"Failed to refresh index stats, must exit: {err}")

    def query(self, input_vector: list[float], top_k=3):
        results = self.index.query(
            namespace=self.namespace,
            vector=input_vector,
            top_k=top_k,
            include_values=False,
            include_metadata=True,
            show_progress=True,
        )

        return results

    def return_sources(self):
        """
        Returns the sources from metadata of the vector embeddings 

        Args:
            no args
        Returns:
            list: each item is a tuple with the source string and the number of vectors generated and stored from the doc
        """
        if self.cached_vectors_count == 0:
            return None
        source_list = Counter(metadata.get("metadata").get("source", "Unknown Source") for _, metadata in self.cached_vectors.items())
        return sorted(source_list.items())

    def refresh_cache(self):
        ru = 0

        for ids in self.index.list(limit=self.max_batch_size, namespace=self.namespace):
            fetch_response = self.index.fetch(ids=ids, namespace=self.namespace)
            ru += int(fetch_response.get("usage", {}).get("read_units", 0))

            for items in fetch_response["vectors"].items():
                self.cached_vectors[items[0]] = {"metadata": items[1]["metadata"]}

        print(f"{ru} read units used for this operation")

        self.read_units_used += ru

        print("Trying to serialize the vectors to cache.")
        directory = os.path.dirname(self.local_cache)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(self.local_cache, "wb") as file:
            pickle.dump(self.cached_vectors, file)

        print("Successfully pickled the vectors.")
        self.cached_vectors_count = len(self.cached_vectors)
        self.cache_refreshed_already = True

    def _cache_synced(self):
        """
        The function checks if we have a local replica, and loads the self.all_vectors with it, otherwise does not do anything
        """
        print("checking cache...")

        cache_fully_synced = False

        if os.path.exists(self.local_cache):
            try:
                with open(self.local_cache, "rb") as file:
                    self.cached_vectors = pickle.load(file)
                    self.cached_vectors_count = len(self.cached_vectors)
                    if len(self.cached_vectors) == 0:
                        print("vector file read successfully, but there are no vectors in it :(")
                        cache_fully_synced = False
                    elif self.cached_vectors_count != self.ns_vectorcount:
                        print("we have a different amount of vectors in the cache than actually in the namespace..")
                        print(f"ns: {self.ns_vectorcount} vs cache: {self.cached_vectors_count}")
                        cache_fully_synced = False
                    else:
                        self.cached_vectors_count = len(self.cached_vectors)
                        print(f"vector dict file read successfully, we have all the {self.cached_vectors_count} vectors in the cache as well.")
                        cache_fully_synced = True
            except KeyError as ke:
                print(f"The cache file is unreadable, continuing. Error: {ke}")
                cache_fully_synced = False
            except Exception as e:
                print(f"An error occurred during reading the vector file: {e}")
                cache_fully_synced = False
        else:  # os.path. not exists(self.local_cache)
            cache_fully_synced = False

        return cache_fully_synced
    

def process_pc_qr(pinecone_response, mss: float) -> str | None:
    """
    Processes relevant vectors from Pinecone and extracts useful metadata fields
    to create a structured context string.
    """
    relevant_vectors = [match for match in pinecone_response.matches if match.score > mss]

    if not relevant_vectors:
        return None

    context_str_list = []

    for i, match in enumerate(relevant_vectors, start=1):
        metadata = match.metadata
        score = match.score
        
        # Define important fields to extract
        important_fields = ["title", "main_header", "description", "header_0", "header_1", "header_2", "content"]
        extracted_fields = {key: metadata.get(key, "N/A") for key in important_fields}
        
        source = metadata.get("source", "Unknown")
        
        # Construct the context string
        context_str = (f"#### Context {i} BEGIN ####\n"
                       f"Title: {extracted_fields['title']}\n"
                       f"Main Header: {extracted_fields['main_header']}\n"
                       f"Description: {extracted_fields['description']}\n"
                       f"Header 0: {extracted_fields['header_0']}\n"
                       f"Header 1: {extracted_fields['header_1']}\n"
                       f"Header 2: {extracted_fields['header_2']}\n"
                       f"Content:\n{extracted_fields['content']}\n\n"
                       f"Source: {source}\n"
                       f"Score: {score:.5f}\n"
                       f"#### Context {i} END ####\n")

        context_str_list.append(context_str)

    return "\n".join(context_str_list)


def get_pinecone_client():
    """Instantiate a Pinecone client object."""
    return PineConeClient()
