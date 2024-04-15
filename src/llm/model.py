from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer

from src.data_mod import TeachData

class TeachModelLM:
    def __init__(self, model_name="gpt2", UTT=True, ST=False, DH=False, DA_E=False, data=None):
        if data is None:
            data = TeachData(model_name, UTT=UTT, ST=ST, DH=DH, DA_E=DA_E)
        pass