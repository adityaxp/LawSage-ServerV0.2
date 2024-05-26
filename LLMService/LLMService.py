import uvicorn
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from logger import logtool
from ChatCompletionResponse import query_response
 

class ConnectionItem(BaseModel):
    auth_token: str


class QueryItem(BaseModel):
    auth_token: str
    prompt: str


app = FastAPI()
deviceName = torch.cuda.get_device_name(0)
resp = []


@app.post("/LAW-SAGE")
async def LAW_SAGE_llamacpp_request(query_item: QueryItem):
    logtool.write_log(f"Fetching response...", "LLM-Service")
    if query_item.auth_token == "ISAUodiuIAU21":
        result = query_response(resp , query_item.prompt)
        resp.append({"role": "user", "content": query_item.prompt})
        resp.append({"role": "assistant", "content": result["content"]})
        return {
            "prompt": query_item.prompt,
            "response" : result
        }
    else:
        return {"response": "Auth failed!"}
    


    
@app.post("/LAW-SAGE-TEST")
async def LAW_SAGE_request(query_item: QueryItem):
    if query_item.auth_token == "ISAUodiuIAU21":
        return {
            "prompt": query_item.prompt,
            "response" : "TEST RESPONSE"
        }
    else:
        return {"message": "Auth failed!"}


@app.get("/CHECK", status_code=200)
async def read_root():
    return {
        "connection": True,
        "deviceName": deviceName, 
        "message": "Connection successful!"}


if __name__ == '__main__':
    uvicorn.run(app, port=8001, host='192.168.1.8')
