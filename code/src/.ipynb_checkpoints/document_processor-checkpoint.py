import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor(object):
    def __init__(self, original_docs) -> None:
        self.orignal_doc_lis = original_docs
        self.docs = self.process_docs(original_docs)

    def web_loader(self, doc_lis):
        loader = WebBaseLoader(
                web_paths=doc_lis,
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("hero__text-section-container col col-12 col-lg-6", "site-main")
                    )
                ),
            )
        docs = loader.load()
        return docs
    
    def split_text(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        return text_splitter.split_documents(docs)
        
    def process_docs(self,docs):
        docs = self.web_loader(docs)
        docs = self.split_text(docs)
        return docs
