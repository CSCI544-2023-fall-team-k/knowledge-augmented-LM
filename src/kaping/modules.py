import dspy

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    context = dspy.InputField(desc="facts that might be relevant to answer the question")
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()