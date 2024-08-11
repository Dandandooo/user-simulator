from src.model.llms import AzureLM
import os

api_key = os.getenv("AZURE_OPENAI_KEY_4o")
model = "gpt-4o-mini"
endpoint = "https://uiuc-convai.openai.azure.com/"

client = AzureLM(model, api_key, endpoint)

EPOCHS = 1
dataset_name = "0-shot_100pc-obs"

fine_tune_id = client.submit_finetune(dataset_name, epochs=EPOCHS)

print("\x1b[35;1mFine-tuning ID:\x1b[0m", fine_tune_id)


