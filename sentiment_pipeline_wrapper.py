import pandas as pd
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
class SentimentPipelineWrapper:
  def __init__(self,model_name = "cardiffnlp/twitter-roberta-base-sentiment"):
    #from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    self.model_name=model_name
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(model_name,device_map=None,low_cpu_mem_usage=False).to("cpu")
    self.sentiment_model = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=-1)

  def analyze_sentiments(self,review_list):
    if not review_list:
      return []
    results = self.sentiment_model(review_list)
    return [(review, res['label'], res['score']) for review, res in zip(review_list, results)]
  
  def split_reviews(self,text):
    if not isinstance(text, str) or text.strip().lower() == "No reviews available":
      return []
    return [r.strip() for r in text.split("\n\n") if r.strip()]
