import os

from openai import OpenAI


LITELLM_BASE_URL = "https://litellm.service-gateway.dev.imw.fraunhofer.de"
MODEL_NAME = "gpt-4.1-mini"


def main() -> None:
    api_key = os.getenv("LITELLM_API_KEY")
    if not api_key:
        raise ValueError("Set LITELLM_API_KEY before running this smoke test.")

    client = OpenAI(
        api_key=api_key,
        base_url=LITELLM_BASE_URL,
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "hey"}],
        max_tokens=16,
        temperature=0,
    )

    choice = response.choices[0]
    print("model:", response.model)
    print("finish_reason:", choice.finish_reason)
    print("message:", choice.message.content)


if __name__ == "__main__":
    main()
