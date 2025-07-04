from sentence_transformers import SentenceTransformer
import pymongo
from dotenv import load_dotenv
import os

load_dotenv()   

mongo_uri = os.getenv("MONGO_URI")
client = pymongo.MongoClient(mongo_uri)
db = client.sample_mflix
collection = db.movies
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # dimension 384

def generate_embedding(sentences: str) -> list[float]:
    embeddings = model.encode(sentences, normalize_embeddings=False).tolist()
    # print(embeddings)
    return embeddings
    
    # Uncomment the following lines if you want to use OpenAI's embedding model
    # response = openai.Embedding.create(
    #     model="text-embedding-ada-002",
    #     input=text
    # )
    # return response['data'][0]['embedding']

query = "rich young Easterner"

# Uncomment the following lines to generate embeddings for existing documents
# for doc in collection.find({'plot':{"$exists": True}}).limit(50):
#   doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
#   collection.replace_one({'_id': doc['_id']}, doc)

results = list(collection.aggregate([
    {
        "$search": {
            "index": "plotSemanticSearch",
            "knnBeta": {
                "vector": generate_embedding(query),
                "path": "plot_embedding_hf",
                "k": 10
            }
        },
        # Uncomment these lines if you want to use vector search
        #   "$vectorSearch": {
        #     "queryVector": generate_embedding(query),   
        #     "path": "plot_embedding_hf",
        #     "numCandidates": 100,
        #     "limit": 4,
        #     "index": "plotSemanticSearch"
        #   }              
    },
    {
        "$project": {
            "title": 1,
            "plot": 1,
            "score": {"$meta": "searchScore"}
        }
    },
      {
        "$match": {
            "score": {"$gte": 0.6}          # Uncomment this line if you want to use vector search
        }
    },
    {
        "$limit": 10  # final limit AFTER filtering
    }
]))

results_list = list(results)
print(f"Number of results: {len(results_list)}")

for doc in results_list:
    print(f'Movie Name: {doc.get("title")},\nMovie Plot: {doc.get("plot")}   score:{doc.get("score")}\n')
