from langchain.memory import ConversationBufferMemory

class MemoryService:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
