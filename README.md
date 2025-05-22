
# Aim:
Development of Python Code Compatible with Multiple AI Tools

# Algorithm: 
Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights.

# Objective
To automate communication with multiple AI APIs using Python.

To standardize and compare outputs from different AI models on a common task or dataset.

To analyze performance or response quality and extract actionable insights.

To demonstrate a unified interface for AI benchmarking and decision support.

# Algorithm
1. API Setup: Configure authentication for each AI service (store API keys securely).

2. Prompt Input: Define a set of prompts or tasks to test.

a) API Request Loop:

b) For each AI tool, send the same prompt.

c) Collect responses.

3. Output Normalization: Preprocess or clean outputs for consistency.

a) Comparison Logic:

Use semantic similarity, token length, keyword match, or sentiment analysis.

b) Insight Generation:

4. Rank responses or summarize strengths and weaknesses.

# Reporting:

Output comparison report in CSV or Markdown.

Optionally use visualization (matplotlib/seaborn).

# Required Libraries
```
pip install openai cohere anthropic requests pandas numpy nltk matplotlib seaborn
```
# Procedure
1. Setting Up API Access
OpenAI: Sign up at https://platform.openai.com/ → get API key.

Cohere: Sign up at https://dashboard.cohere.ai/ → generate API key.

Anthropic (Claude): Apply for access at https://www.anthropic.com/ and get API key.

Hugging Face: Get token from https://huggingface.co/settings/tokens.

Store these API keys in a .env file for security.

2. Step-by-Step Python Code
a) Environment Setup (.env)
```
OPENAI_API_KEY=your_openai_key
COHERE_API_KEY=your_cohere_key
ANTHROPIC_API_KEY=your_anthropic_key
```
b) Python Script
```
import os
import openai
import cohere
import anthropic
from dotenv import load_dotenv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
co = cohere.Client(os.getenv("COHERE_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Define a prompt
prompt = "Explain the benefits of using renewable energy sources."

# Get OpenAI response
def get_openai_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Get Cohere response
def get_cohere_response(prompt):
    response = co.generate(prompt=prompt, model="command", max_tokens=300)
    return response.generations[0].text.strip()

# Get Anthropic response
def get_claude_response(prompt):
    response = anthropic_client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# Collect responses
responses = {
    "OpenAI GPT-4": get_openai_response(prompt),
    "Cohere": get_cohere_response(prompt),
    "Claude (Anthropic)": get_claude_response(prompt)
}

# Display responses
for tool, reply in responses.items():
    print(f"\n--- {tool} ---\n{reply}\n")

# Similarity Comparison
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(responses.values())
similarity_matrix = cosine_similarity(vectors)

# Display similarity matrix
tools = list(responses.keys())
sim_df = pd.DataFrame(similarity_matrix, index=tools, columns=tools)
print("\nSemantic Similarity Matrix:\n", sim_df.round(2))

# Generate insights
def generate_insight(similarity_df):
    insights = []
    for tool in similarity_df.columns:
        most_similar = similarity_df[tool].drop(tool).idxmax()
        score = similarity_df[tool].drop(tool).max()
        insights.append(f"{tool} is most similar to {most_similar} (score: {score:.2f})")
    return insights

print("\nActionable Insights:")
for insight in generate_insight(sim_df):
    print("•", insight)
```
# Output (Example)
```
--- OpenAI GPT-4 ---
Renewable energy sources offer numerous benefits, such as reduced greenhouse gas emissions...

--- Cohere ---
The main advantages of renewable energy include sustainability, cost-effectiveness...

--- Claude (Anthropic) ---
Using renewable energy reduces environmental impact, lowers long-term costs...

Semantic Similarity Matrix:
                   OpenAI GPT-4  Cohere  Claude (Anthropic)
OpenAI GPT-4              1.00    0.89               0.92
Cohere                    0.89    1.00               0.87
Claude (Anthropic)        0.92    0.87               1.00

Actionable Insights:
• OpenAI GPT-4 is most similar to Claude (Anthropic) (score: 0.92)
• Cohere is most similar to OpenAI GPT-4 (score: 0.89)
• Claude (Anthropic) is most similar to OpenAI GPT-4 (score: 0.92)
```
# Conclusion
This Python-based automation system successfully integrates multiple AI APIs to standardize input prompts, retrieve model outputs, compare them semantically, and produce actionable insights. The solution can be extended further to support bulk prompts, visualization dashboards, or automated reporting pipelines. It provides a valuable framework for benchmarking AI tools, conducting experiments, or even choosing the most suitable LLM for specific tasks in production environments.

# Result: 
The corresponding Prompt is executed successfully
