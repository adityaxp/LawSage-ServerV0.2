from llama_cpp import Llama
from logger import logtool
from RAG_central_acts import RAG_central_acts_get_context
from RAG_constitution import RAG_constitution_get_context
from RAG_state_acts import RAG_state_acts_get_context

llama2_chat = Llama(model_path="./models/mistral-7b-instruct-v0.1.Q3_K_S.gguf", 
                    n_gpu_layers=5, 
                    n_threads=3, 
                    n_ctx=20000,
                    repeat_penalty=1.1,
                    temperature=0.8,
                    top_p=0.95, 
                    verbose=True) 

system_prompt = """
You are a helpful, respectful and honest AI powered legal advisory assistant named Law Sage.
Always answer as helpfully as possible, while being safe.
Additionally, please provide the appropriate references for legal queries.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please follow the provided context to generate a response, and do not use any other information.
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information.
"""

RESPONSE_MAX_TOKENS = 550  
resp = []

def query_response(previous_chats, prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        *previous_chats,
        {"role": "user", "content": prompt}
    ]
    
    resp = llama2_chat.create_chat_completion(messages=messages, max_tokens=RESPONSE_MAX_TOKENS)
    return resp["choices"][0]["message"]["content"]  

def get_RAG_response(query, RAG_type):
    logtool.write_log("Generating RAG response", "RAG")
    context = []
    if RAG_type == "RAG_constitution":
        context = RAG_constitution_get_context(query)
        query = query + " Context: " + str(context[0])
    elif RAG_type == "RAG_central_acts":
        context = RAG_central_acts_get_context(query)
        query = query + " Context: " + str(context[0]) 
    elif RAG_type == "RAG_state_acts":
        context = RAG_state_acts_get_context(query)
        query = query + " Context: " + str(context[0])
    else:
        return "Error: Undefined RAG Type"
      
    result = query_response(resp, query)
    resp.append({"role": "user", "content": query})
    resp.append({"role": "assistant", "content": result})


    return result

