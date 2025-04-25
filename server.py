from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM
from mcp.server.fastmcp import FastMCP
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and tokenizer
logger.info("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(
    "./llama3.2_1b_local", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    "./llama3.2_1b_local", device_map="auto")
logger.info("Model loaded successfully.")

# Initialize MCP
mcp = FastMCP(name="LLaMA 3.2-1B")

# Define tool


@mcp.tool(name="generate_text", description="Generate text from a prompt")
def generate_text(prompt: str) -> str:
    logger.info(f"Generating response for prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info("Generated response.")
    return response


# Create FastAPI app
app = FastAPI()


class PromptInput(BaseModel):
    prompt: str


@app.post("/generate_text")
async def query_llama(data: PromptInput):
    logger.info(f"Received request with prompt: {data.prompt}")
    result = generate_text(data.prompt)
    logger.info("Returning response to client.")
    return {"response": result}

# Entry point
if __name__ == "__main__":
    logger.info("Starting HTTP server with MCP tool...")
    uvicorn.run(app, host="0.0.0.0", port=5001)
