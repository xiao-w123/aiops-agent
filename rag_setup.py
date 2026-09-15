from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. 加载 knowledge 文件夹下的所有 txt 文档
loader = DirectoryLoader(
    "knowledge/",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
docs = loader.load()
print(f"加载了 {len(docs)} 篇文档")

# 2. 切分文档
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", " "],
)
chunks = splitter.split_documents(docs)
print(f"切分成 {len(chunks)} 个片段")

# 3. 用本地 Embedding 模型向量化
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 4. 存入 Chroma
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db",
)
print("知识库建好了，存在 ./chroma_db")