from llama_cpp import Llama

llama2_chat = Llama(model_path="./models/llama-3-8b-instruct-law-sage-v0.1.Q4_K_M.gguf", 
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

MAX_TOKENS = 1024  
RESPONSE_TOKENS = 512  

def truncate_conversation(conversation, max_tokens):
    total_tokens = 0
    truncated_conversation = []
    for message in reversed(conversation):
        message_tokens = len(message["content"].split())
        if total_tokens + message_tokens <= max_tokens:
            truncated_conversation.append(message)
            total_tokens += message_tokens
        else:
            break
    return list(reversed(truncated_conversation))

def chat(previous_chats, prompt):
    #truncated_chats = truncate_conversation(previous_chats, MAX_TOKENS - RESPONSE_TOKENS)
    messages = [
        {"role": "system", "content": system_prompt},
        *previous_chats,
        {"role": "user", "content": prompt}
    ]
    
    resp = llama2_chat.create_chat_completion(messages=messages, max_tokens=RESPONSE_TOKENS)
    return resp["choices"][0]["message"]

resp = []
conv = 0

while True: 
    prompt = input("> ")
    print(f"Conversation {conv+1}:")
    response = chat(resp, prompt)
    
    resp.append({"role": "user", "content": prompt})
    resp.append({"role": "assistant", "content": response["content"]})
    
    print("LawSage: ", response["content"])
    conv += 1