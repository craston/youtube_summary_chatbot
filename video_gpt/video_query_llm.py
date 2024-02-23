from typing import Any

from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

from langchain_community.vectorstores.faiss import FAISS

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
    def __init__(self, youtube_url:str) -> None:
        self.youtube_url = youtube_url
        self.transcript, self.retriever = self.generate_retriever_from_videourl()
        self._init_prompts_templates()
        self.memory = ConversationBufferMemory(
            return_messages=True, output_key="answer", input_key="question"
        )
        self.standalone_question, self.final_chain = self.create_chain()
        self.summary_chain = load_summarize_chain(ChatOpenAI(), chain_type="map_reduce")

    def generate_retriever_from_videourl(self) -> VectorStoreRetriever:
        transcript_loader = YoutubeLoader.from_youtube_url(self.youtube_url, language="en")
        transcript = transcript_loader.load_and_split()
        oai_embedding_model = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(transcript, oai_embedding_model)
        faiss_retriever = vector_store.as_retriever()
        return transcript, faiss_retriever

    
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

        self.DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

    def _combine_documents(
        self, docs, document_prompt, document_separator="\n\n"
        ):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    def create_chain(self):
        # First we add a step to load memory
        # This adds a "memory" key to the input object
        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("history"),
        )
        # Now we calculate the standalone question
        standalone_question = {
            "standalone_question": {
                "question": lambda x: x["question"],
                "chat_history": lambda x: get_buffer_string(x["chat_history"]),
            }
            | self.CONDENSE_QUESTION_PROMPT
            | ChatOpenAI(temperature=0)
            | StrOutputParser(),
        }

        # Now we retrieve the documents
        retrieved_documents = {
            "docs": itemgetter("standalone_question") | self.retriever,
            "question": lambda x: x["standalone_question"],
        }

        # Now we construct the inputs for the final prompt
        final_inputs = {
            "context": lambda x: self._combine_documents(docs= x["docs"], document_prompt=self.DEFAULT_DOCUMENT_PROMPT),
            "question": itemgetter("question"),
        }
        # And finally, we do the part that returns the answers
        answer = {
            "answer": final_inputs | self.ANSWER_PROMPT | ChatOpenAI(), 
            "docs": itemgetter("docs"),
        }
        # And now we put it all together!
        final_chain = loaded_memory | standalone_question | retrieved_documents | answer
        return standalone_question, final_chain 
