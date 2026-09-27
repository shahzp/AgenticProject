from deepeval.metrics import FaithfulnessMetric,AnswerRelevancyMetric,ContextualPrecisionMetric,ContextualRecallMetric


def get_metrics(model):
    context_precision_metric = ContextualPrecisionMetric(threshold=0.7, model=model, include_reason=True)
    context_recall_metric = ContextualRecallMetric(threshold=0.7, model=model,include_reason=True)
    # answer_relevancy_metric= AnswerRelevancyMetric(threshold=0.7, model=model,include_reason=True)
    # faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=model,include_reason=True)
    evaluation_metrics = [context_recall_metric,context_precision_metric]
    return evaluation_metrics