import requests


def main():
    prompt = input("Enter your prompt: ")
    url = "http://0.0.0.0:5001/generate_text"  # MCP default port + tool name
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json={"prompt": prompt}, headers=headers)

    if response.status_code == 200:
        print("Model Response:", response.json()["response"])
    else:
        print("Error:", response.status_code)
        print("Message:", response.text)


if __name__ == "__main__":
    main()
