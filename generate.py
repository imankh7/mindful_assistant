import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from chunk import chunk_document

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

GENERATION_PROPMT = """ You are supportive assistant who is answering users' questions about mental health ONLY with provided context.

Answer users' question with only using context information bellow.
If the context does not containt the information for the answer, say you don't have enough information to answer.
Do not add the information that is not in the context. 

Context:
{context}"""


def generate_answer(question, retriever):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = GENERATION_PROPMT.format(context=context, question=question)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = chunk_document("data/processed/gad.json")
    store = FAISS.from_documents(documents, embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 3})

    question = "What are the symptoms of generalized anxiety disorder?"
    answer = generate_answer(question, retriever)

    print(f"Question: {question}\n")
    print(f"Answer:\n{answer}")
