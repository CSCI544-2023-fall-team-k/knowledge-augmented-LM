import dspy

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    context = dspy.InputField(desc="facts that might be relevant to answer the question")
    answer = dspy.OutputField(desc="often between 1 and 5 words")