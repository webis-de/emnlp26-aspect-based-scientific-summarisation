"""One-off probe: print Azure OpenAI's rate-limit headers for the GPT-4.1-mini deployment.

Not part of the pipeline. Run once, read the numbers, delete/ignore afterward.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("GPT_4_1_MINI_AZURE_ENDPOINT"),
    api_key=os.getenv("GPT_4_1_MINI_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("GPT_4_1_MINI_API_VERSION"),
    max_retries=0,
)
deployment = os.getenv("GPT_4_1_MINI_DEPLOYMENT_NAME")

raw = client.chat.completions.with_raw_response.create(
    model=deployment,
    messages=[{"role": "user", "content": "Reply with the single word: ok"}],
    temperature=0,
    max_tokens=1,
)

headers = raw.headers
print("=== Response headers of interest ===")
for key in headers.keys():
    if "ratelimit" in key.lower() or key.lower() in ("x-request-id", "apim-request-id"):
        print(f"{key}: {headers.get(key)}")

print("\n=== All headers (fallback if none matched above) ===")
for key, value in headers.items():
    print(f"{key}: {value}")
