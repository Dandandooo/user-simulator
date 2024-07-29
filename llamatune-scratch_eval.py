from src.model.llms import HugLM

dataset = "0_no_move_20pc_obs"

model_name = f"llm_models/user-sim__llama-3-8b-it__{dataset}"

llama3 = HugLM(model_name)
llama3.data.load(dataset)

if __name__ == '__main__':
    llama3.save_answers(dataset, "test", f"llm_prompt_sessions/llama_train/{dataset}.json")