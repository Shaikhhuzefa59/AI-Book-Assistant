from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = HuggingFaceEmbeddings()

vectorestore = Chroma(
    persist_directory='chroma_db',
    embedding_function = embedding_model
)

retriever = vectorestore.as_retriever(
    search_type='mmr',
    search_kwargs = {
        'k':4,
        'fetch_k':10,
        "lambda_mult":0.5
    }
)

llm = ChatMistralAI(model = 'mistral-small-2506')

prompt = ChatPromptTemplate.from_messages([
    ('system',
     """You are a helpful Ai Assitant.
     use only the provided context to answer the questions.
     if the answer is not present in the context, say: "I Could not find the answer in the document."
        """),
    ('human',
     """Context:
     {context}
     Question:
     {question}""")
])

print('Rag System Created')

print('press 0 to Exit')

while True:
    query = input('You :')
    if query==0:
        break

    docs = retriever.invoke(query)
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        'context':context,
        'question':query
    })

    response = llm.invoke(final_prompt)
    print(f'\n AI : {response.content}')
