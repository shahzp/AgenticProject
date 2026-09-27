from rbiFaqAgent.Evals.GroqJudge import GroqJudge
class EvaluationJudge:

    def __init__(self):
        self.model=GroqJudge()

    def get_evaluation_model(self):
        return self.model
