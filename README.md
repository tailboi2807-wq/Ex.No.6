# Ex.No.6 Development of Python Code Compatible with Multiple AI Tools

# Date:
# Register no.
# Aim: Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights with Multiple AI Tools

#AI Tools Required:

# Explanation:
Experiment the persona pattern as a programmer for any specific applications related with your interesting area. 
Generate the outoput using more than one AI tool and based on the code generation analyse and discussing that. 

# Conclusion:
Objective
To develop a Python-based AI Integration Framework that:
Connects to multiple AI providers.
Executes a prompt or task across all of them.
Returns results in a standardized format for comparison or reporting.

⚙️ Supported AI Tools
AI Tool	Provider	Library Used	Example Model
OpenAI	ChatGPT, GPT-4	openai	gpt-4o-mini
Anthropic	Claude	anthropic	claude-3-sonnet
Google AI	Gemini	google-generativeai	gemini-1.5-flash
Hugging Face	Transformers API	requests	google/flan-t5-xl

Python Code: Multi-AI Integration Framework

multi_ai_framework.py
A Python framework compatible with multiple AI tools.
Developed for testing, comparison, and report generation across AI platforms.
"""

import os
import requests

# Import optional SDKs (install only what you need)
import openai
import anthropic
import google.generativeai as genai

# -----------------------------
# Configuration (Set API keys)
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# -----------------------------
# AI Interface Functions
# -----------------------------
def query_openai(prompt: str, model="gpt-4o-mini"):
    """Query OpenAI GPT models"""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"OpenAI Error: {e}"

def query_anthropic(prompt: str, model="claude-3-sonnet"):
    """Query Anthropic Claude models"""
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Anthropic Error: {e}"

def query_gemini(prompt: str, model="gemini-1.5-flash"):
    """Query Google Gemini models"""
    try:
        model = genai.GenerativeModel(model)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Google Gemini Error: {e}"

def query_huggingface(prompt: str, model="google/flan-t5-xl"):
    """Query Hugging Face models via API"""
    try:
        API_URL = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        payload = {"inputs": prompt}
        response = requests.post(API_URL, headers=headers, json=payload)
        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return str(data)
    except Exception as e:
        return f"Hugging Face Error: {e}"

# -----------------------------
# Unified Comparison Function
# -----------------------------
def compare_ai_responses(prompt: str):
    """Generate and compare responses from all configured AI tools"""
    print("\n🔹 Input Prompt:", prompt)
    print("-" * 70)

    results = {}

    # Run across multiple AI tools
    results["OpenAI GPT"] = query_openai(prompt)
    results["Anthropic Claude"] = query_anthropic(prompt)
    results["Google Gemini"] = query_gemini(prompt)
    results["Hugging Face"] = query_huggingface(prompt)

    # Display Results
    for tool, output in results.items():
        print(f"\n🧩 {tool} Response:\n{output}\n{'-'*70}")

    return results

# -----------------------------
# Example Execution
# -----------------------------
if __name__ == "__main__":
    test_prompt = "Explain blockchain technology in simple terms suitable for school students."
    compare_ai_responses(test_prompt)

Sample Output

🔹 Input Prompt: Explain blockchain technology in simple terms suitable for school students.
----------------------------------------------------------------------
🧩 OpenAI GPT Response:
Blockchain is like a digital notebook that everyone can see but no one can erase...

🧩 Anthropic Claude Response:
Imagine a shared online diary where every page is locked once written...

🧩 Google Gemini Response:
Blockchain is a secure digital record that helps people trust online transactions...

🧩 Hugging Face Response:
A blockchain is a chain of data blocks linked together that ensures security...

🧱 Advantages of Multi-AI Code Development
✅ Cross-Model Benchmarking: Compare reasoning, creativity, and factual accuracy.
✅ Unified Testing Environment: Use one framework to test prompt performance.
✅ Supports Report Generation: Ideal for prompt engineering research projects.
✅ Reusability: Easily integrated into GUI apps or data pipelines.

🔒 
Security & API Key Management
Always store API keys as environment variables (never hardcode).
Use .env files or a secure key vault.
Example for Linux/macOS:
##output
🧩 Objective
To develop a Python-based AI Integration Framework that:
Connects to multiple AI providers.
Executes a prompt or task across all of them.
Returns results in a standardized format for comparison or reporting.

⚙️ Supported AI Tools
AI Tool	Provider	Library Used	Example Model
OpenAI	ChatGPT, GPT-4	openai	gpt-4o-mini
Anthropic	Claude	anthropic	claude-3-sonnet
Google AI	Gemini	google-generativeai	gemini-1.5-flash
Hugging Face	Transformers API	requests	google/flan-t5-xl

Python Code: Multi-AI Integration Framework

multi_ai_framework.py
A Python framework compatible with multiple AI tools.
Developed for testing, comparison, and report generation across AI platforms.
"""

import os
import requests

# Import optional SDKs (install only what you need)
import openai
import anthropic
import google.generativeai as genai

# -----------------------------
# Configuration (Set API keys)
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# -----------------------------
# AI Interface Functions
# -----------------------------
def query_openai(prompt: str, model="gpt-4o-mini"):
    """Query OpenAI GPT models"""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"OpenAI Error: {e}"

def query_anthropic(prompt: str, model="claude-3-sonnet"):
    """Query Anthropic Claude models"""
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Anthropic Error: {e}"

def query_gemini(prompt: str, model="gemini-1.5-flash"):
    """Query Google Gemini models"""
    try:
        model = genai.GenerativeModel(model)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Google Gemini Error: {e}"

def query_huggingface(prompt: str, model="google/flan-t5-xl"):
    """Query Hugging Face models via API"""
    try:
        API_URL = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        payload = {"inputs": prompt}
        response = requests.post(API_URL, headers=headers, json=payload)
        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return str(data)
    except Exception as e:
        return f"Hugging Face Error: {e}"

# -----------------------------
# Unified Comparison Function
# -----------------------------
def compare_ai_responses(prompt: str):
    """Generate and compare responses from all configured AI tools"""
    print("\n🔹 Input Prompt:", prompt)
    print("-" * 70)

    results = {}

    # Run across multiple AI tools
    results["OpenAI GPT"] = query_openai(prompt)
    results["Anthropic Claude"] = query_anthropic(prompt)
    results["Google Gemini"] = query_gemini(prompt)
    results["Hugging Face"] = query_huggingface(prompt)

    # Display Results
    for tool, output in results.items():
        print(f"\n🧩 {tool} Response:\n{output}\n{'-'*70}")

    return results

# -----------------------------
# Example Execution
# -----------------------------
if __name__ == "__main__":
    test_prompt = "Explain blockchain technology in simple terms suitable for school students."
    compare_ai_responses(test_prompt)

Sample Output

🔹 Input Prompt: Explain blockchain technology in simple terms suitable for school students.
----------------------------------------------------------------------
🧩 OpenAI GPT Response:
Blockchain is like a digital notebook that everyone can see but no one can erase...

🧩 Anthropic Claude Response:
Imagine a shared online diary where every page is locked once written...

🧩 Google Gemini Response:
Blockchain is a secure digital record that helps people trust online transactions...

🧩 Hugging Face Response:
A blockchain is a chain of data blocks linked together that ensures security...

🧱 Advantages of Multi-AI Code Development
✅ Cross-Model Benchmarking: Compare reasoning, creativity, and factual accuracy.
✅ Unified Testing Environment: Use one framework to test prompt performance.
✅ Supports Report Generation: Ideal for prompt engineering research projects.
✅ Reusability: Easily integrated into GUI apps or data pipelines.

🔒 
Security & API Key Management
Always store API keys as environment variables (never hardcode).
Use .env files or a secure key vault.
Example for Linux/macOS:

# Result: The corresponding Prompt is executed successfully.
