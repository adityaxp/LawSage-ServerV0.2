from llama_cpp import Llama


lawsage_chat = Llama(model_path="./models/llama-3-8b-instruct-law-sage-v0.1.Q4_K_M.gguf", 
                    n_gpu_layers=20, 
                    n_threads=6, 
                    n_ctx=3584, 
                    n_batch=521, 
                    verbose=True) 


system_prompt = """
You are a helpful, respectful and honest AI powered legal advisory assistant named Law Sage.
Always answer as helpfully as possible, while being safe.
Additionally, please provide the appropriate references for legal queries.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information.
"""

RESPONSE_MAX_TOKENS = 550  

def query_response(previous_chats, prompt):
    #truncated_chats = truncate_conversation(previous_chats, MAX_TOKENS - RESPONSE_TOKENS)
    messages = [
        {"role": "system", "content": system_prompt},
        *previous_chats,
        {"role": "user", "content": prompt}
    ]
    
    resp = lawsage_chat.create_chat_completion(messages=messages, max_tokens=RESPONSE_MAX_TOKENS)
    return resp["choices"][0]["message"]
