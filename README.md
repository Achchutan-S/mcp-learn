---
noteId: "5fd20f7021b311f0a308f5d26d63541a"
tags: []

---

# LLaMA 3.2-1B MCP Server

This project implements a server for the LLaMA 3.2-1B model using the Hugging Face `transformers` library and the MCP (Model Context Protocol) framework. The server provides an API to generate text based on a given prompt.

## Features

- **Text Generation**: Generate text using the `generate_text` tool.
- **FastAPI Integration**: Exposes an HTTP endpoint for text generation.
- **MCP Tooling**: Supports MCP tools for seamless integration.
- **Local Model Loading**: Loads the LLaMA model from a local directory for offline use.

---

## Requirements

- Python 3.8 or higher
- Virtual environment (recommended)

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <your-repository-url>
   cd python-mcp
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv mcp_env
   source mcp_env/bin/activate
   ```

3. **Install Dependencies**:
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Model**:
   Ensure the LLaMA 3.2-1B model is downloaded and saved in the `./llama3.2_1b_local` directory.

---

## Usage

### Run the Server

1. **Using the Terminal**:
   Start the server with:
   ```bash
   python server.py
   ```

2. **Using VS Code**:
   - Open the project in VS Code.
   - Go to the "Run and Debug" panel.
   - Select the `Python: Run Server` configuration and click "Start Debugging".

### API Endpoint

- **POST /generate_text**  
  Generate text based on a given prompt.

  **Request**:
  ```json
  {
    "prompt": "What is the capital of France?"
  }
  ```

  **Response**:
  ```json
  {
    "response": "The capital of France is Paris."
  }
  ```

---

## Project Structure

```
python-mcp/
├── client.py           # Client script for testing
├── load_llama.py       # Script to load the LLaMA model
├── mcp_server.py       # MCP server implementation
├── requirements.txt    # Python dependencies
├── server_config.json  # Server configuration file
├── server.py           # Main server script
├── llama3.2_1b_local/  # Local directory for the LLaMA model
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── tokenizer.json
└── README.md           # Project documentation
```

---

## Development

### Clear Cache
To clear the Hugging Face cache:
```bash
rm -rf ~/.cache/huggingface
```

### Debugging
Use the VS Code debugger with the provided [`.vscode/launch.json`](.vscode/launch.json) configuration.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.