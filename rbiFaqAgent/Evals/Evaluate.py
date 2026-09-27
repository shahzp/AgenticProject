import json
from rbiFaqAgent.RAG.RAG_chain import RAGChain
from rbiFaqAgent.Evals.metrics import get_metrics
from rbiFaqAgent.Evals.EvaluationJudge import EvaluationJudge
from Utils.rag_settings import evaluation_results_folder
from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import evaluate, DisplayConfig

goldens=[]

def get_golden():
    data=json.load(open('rbiFaqAgent/Evals/Goldens/goldens.json'))
    for data_item in data['test']:
        golden=Golden(input=data_item['input'],
                      expected_output=data_item['expected_output'])
        goldens.append(golden)
    return goldens

def get_testcases(goldens):
    testcases=[]
    invoker = RAGChain()
    for golden in goldens:
        response=invoker.invoke_rag_chain(golden.input)
        test_case=LLMTestCase(
            input=golden.input,
            expected_output=golden.expected_output,
            actual_output=response['answer'],
            retrieval_context=response['context']
        )
        testcases.append(test_case)
    return testcases

def evaluate_testcases():
    goldens=get_golden()
    testcases=get_testcases(goldens)
    evaluationJudge = EvaluationJudge()
    judge_model = evaluationJudge.get_evaluation_model()
    metrics=get_metrics(judge_model)
    evaluate(test_cases=testcases,metrics=metrics,display_config=DisplayConfig(results_folder=evaluation_results_folder))

if __name__=='__main__':
    evaluate_testcases()

