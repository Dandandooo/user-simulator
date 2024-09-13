from src.model.llms import HugLM

dataset = "0_no_move_40pc_obs"
test_dataset = "0_no_move"

model_name = f"llm_models/user-sim__llama-3-8b-it__{dataset}"

llama3 = HugLM(model_name)
llama3.data.load(dataset)
llama3.data.load(test_dataset)

if __name__ == '__main__':
    llama3.save_answers(test_dataset, "test", f"llm_prompt_sessions/llama_train/{dataset}.json")
