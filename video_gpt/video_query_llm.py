from typing import Any

from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

from langchain_community.vectorstores.faiss import FAISS

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.document_loaders import YoutubeLoader
from langchain.schema import format_document
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.summarize import load_summarize_chain


from langchain.memory import ConversationBufferMemory

from operator import itemgetter


class VideoQueryLLM():
    def __init__(self) -> None:
        self.llm = ChatOpenAI(temperature=0)
        self._init_prompts_templates()
        self.memory = ConversationBufferMemory(
            return_messages=True, output_key="answer", input_key="question"
        )

    def load_video(self, youtube_url:str)-> None:
        transcript_loader = YoutubeLoader.from_youtube_url(youtube_url, language="en")
        self.transcript = transcript_loader.load_and_split()
        oai_embedding_model = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(self.transcript, oai_embedding_model)
        self.retriever = vector_store.as_retriever()

    def get_summary(self)-> None:
        
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        return chain.run(self.transcript)
    
    def _init_prompts_templates(self)-> None:
        _template = """Given the following conversation and a follow up question, 
        rephrase the follow up question to be a standalone question.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
        self.CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        self.ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

    @staticmethod
    def combine_documents(documents: list[Document], separator: str = "\n\n") -> str:
        return separator.join(d.page_content for d in documents)

    def create_chain(self):
        # First we add a step to load memory
        # This adds a "memory" key to the input object
        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("history"),
        )

        # Now we calculate the standalone question
        question_with_chat_history = {
            "question_with_chat_history": {
                "question": lambda x: x["question"],
                "chat_history": lambda x: get_buffer_string(x["chat_history"]),
            }
            | self.CONDENSE_QUESTION_PROMPT
            | self.llm
            | StrOutputParser(),
        }
        
        chain = (
            {"context": itemgetter("question_with_chat_history") | self.retriever | self.combine_documents, 
            "question": lambda x: x["question_with_chat_history"],}
            | self.ANSWER_PROMPT
            | self.llm
            | StrOutputParser())


        return loaded_memory| question_with_chat_history | chain
    
    def get_response(self, question: str)-> str:
        chain = self.create_chain()
        response = chain.invoke({"question": question})
        self.memory.save_context({"question": question}, {"answer": response})
        return response
