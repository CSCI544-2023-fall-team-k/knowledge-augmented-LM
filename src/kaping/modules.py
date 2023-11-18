import dspy

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    context = dspy.InputField(desc="facts that might be relevant to answer the question")
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class EvaluateAnswer(dspy.Signature):
    """Given a list of target phrases and a predicted phrase, say "True" if the predicted phrase is semantically identical to one of the target phrases, otherwise say "False".
    Say also "True" if the predicted one is a hypernym or hyponym of targets."""

    target = dspy.InputField(desc="list of target phrases")
    predicted = dspy.InputField(desc="predicted phrase")
    answer = dspy.OutputField(desc="True or False")