from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader('dl-curriculum.pdf')

pdf_doc = loader.load()

print(len(pdf_doc))
print(pdf_doc[0].page_content)