import dspy

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    context = dspy.InputField(desc="facts that might be relevant to answer the question")
    answer = dspy.OutputField(desc="often between 1 and 5 words")

class GenerateFirstSubQuery(dspy.Signature):
    """Want to retrieve an answer for a complex question that might need procedural reasoning. 
    Given the complex question, generate an initial question to solve the complex question."""

    question = dspy.InputField(desc="complex question that involves multiple steps of reasoning.")
    subquery = dspy.OutputField(desc="simple search query that we have to solve first.")
    
class GenerateNextSubQuery(dspy.Signature):
    """Generate a subquery that we need next in order to solve the given complex question. You are given previous subquery and answer pairs. 
The generated subquery should not overlap with the previous subqueries. If the question can be solved using the previous subquery and answer pairs, just return one world, 'NONE'."""

    question = dspy.InputField(desc="complex question that involves multiple steps of reasoning to solve.")
    previous_qa = dspy.InputField(desc="previous subquery answer pairs.")
    subquery = dspy.OutputField(desc="simple next search query.")
    
class SolveQuestion(dspy.Signature):
    """Output an answer of the question using the subquery and answer pairs."""

    question = dspy.InputField(desc="complex question we want to solve.")
    previous_qa = dspy.InputField(desc="context information.")
    answer = dspy.OutputField(desc="often between 1 to 5 words.")

class EvaluateAnswer(dspy.Signature):
    """Given a list of target phrases and a predicted phrase, say "True" if the predicted phrase is semantically identical to one of the target phrases, otherwise say "False".
    Say also "True" if the predicted one is a hypernym or hyponym of targets."""

    target = dspy.InputField(desc="list of target phrases")
    predicted = dspy.InputField(desc="predicted phrase")
    answer = dspy.OutputField(desc="True or False")