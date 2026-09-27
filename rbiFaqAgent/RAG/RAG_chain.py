from langchain_core.prompts import ChatPromptTemplate
from rbiFaqAgent.RAG.Retriever import get_biencoder_retriever
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_groq import  ChatGroq
import os

class RAGChain():
    load_dotenv()
    llm = ChatGroq(model='openai/gpt-oss-120b',
                   temperature=0.3,
                   api_key=os.environ.get('GROQ_API_KEY'))

    def __init__(self):
        self.llm = RAGChain.llm
        self.retriever = get_biencoder_retriever()
        self.prompt=ChatPromptTemplate.from_template(
            """
            You are an RBI policy advisor.Answer the following question using only the provided context.
            Context:{context}
            Question:{query}
            Answer concisely in 100-200 words
            """)

    def get_retrieval_chain(self):
        retrieval_chain = RunnableParallel(context= self.retriever, query= RunnablePassthrough())
        return retrieval_chain

    def invoke_rag_chain(self,query):
        retrieved_result=self.get_retrieval_chain().invoke(query)
        docs=retrieved_result['context']
        #Format retrieved docs for prompt
        context="\n\n".join([doc.page_content for doc in docs])
        #Generate answer
        answer=(self.prompt|self.llm|StrOutputParser()).invoke({'context':context,'query':query})
        return {'answer':answer,'context':[doc.page_content for doc in docs]}



