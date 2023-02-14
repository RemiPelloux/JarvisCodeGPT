class PromptTemplate:
    """A template for generating GPT-3 prompts"""

    def __init__(self, title: str, context: str, question: str):
        self.title = title
        self.context = context
        self.question = question

    def get_prompt(self) -> str:
        """Generates the GPT-3 prompt"""
        prompt = f"{self.title}\n\n{self.context}\n\nQuestion: {self.question}\nRéponse:"
        return prompt

    def get_source_prompt(self) -> str:
        """Generates a GPT-3 prompt to get sources for an answer"""
        prompt = f"{self.title}\n\n{self.context}\n\nQuestion: {self.question}\nSources:"
        return prompt
