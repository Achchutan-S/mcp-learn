from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch

LOCAL_MODEL_DIR = "./llama3.2_1b_local"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


def load_model():
    if os.path.exists(LOCAL_MODEL_DIR):
        print("✅ Loading model from local directory (offline mode).")
        tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_DIR, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_DIR,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
    else:
        print("🌐 Downloading model from Hugging Face and saving locally...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        # Save locally for future use
        tokenizer.save_pretrained(LOCAL_MODEL_DIR)
        model.save_pretrained(LOCAL_MODEL_DIR)
        print("📦 Model saved to local directory.")

    return tokenizer, model


# Load model and tokenizer (from local if available)
tokenizer, model = load_model()

# 🧪 Example usage


prompt = "What is the capital of France?"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n🤖 Model Output:")
print(response)
