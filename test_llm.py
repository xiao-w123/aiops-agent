from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="deepseek-chat")
result = llm.invoke("用一句话解释什么是服务器响应变慢")
print(result.content)