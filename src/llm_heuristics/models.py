import logging
import os
import re

from google import genai
from google.genai import types
from openai import OpenAI


GEMINI_MODELS = {
    "gemini-2.0-flash" : "gemini-2.0-flash-001",
    "gemini-2.0-flash-thinking" : "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.5-flash" : "gemini-2.5-flash",
    "gemini-2.5-pro" : "gemini-2.5-pro",
    "gemma-3-12b" : "gemma-3-12b-it"
}

OPENAI_MODELS = {
    "o1" : "o1-2024-12-17",
    "o3" : "o3-2025-04-16",
    "gpt-4o" : "gpt-4o-2024-08-06",
    "gpt-4.1" : "gpt-4.1-2025-04-14",
    "gpt-5" : "gpt-5-2025-08-07"
}

DEEPSEEK_MODELS = {
    "deepseek-r1" : "deepseek-reasoner",
    "deepseek-v3" : "deepseek-chat",
}

def sanitize_llm_answer(answer) -> str:
    if "```" in answer:
        return re.search(r"```(?:\w+\n)?(.*?)```", answer, re.DOTALL).group(1)
    else:
        return answer


def get_model_id(model, available_models):
    try:
        return available_models[model]
    except KeyError:
        raise ValueError(f"Model {model} is not supported by the chosen framework. Choose one of the following: {list(available_models.keys())}") from None


def run_gemini(model_name, prompt, temperature, top_p):
    logging.info("Retrieving GOOGLE_API_KEY")
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    model_id = get_model_id(model_name, GEMINI_MODELS)

    token_count = client.models.count_tokens(
        model=model_id, contents=prompt
    ).total_tokens
    logging.info(f"#tokens in prompt: {token_count}")

    response = client.models.generate_content(
        model=model_id,
        config=types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            response_mime_type="text/plain",
            thinking_config=types.ThinkingConfig(thinking_budget=-1,
                                                 include_thoughts=True),
        ),
        contents=prompt,
    )
    logging.info("Generating answer...")
    logging.info(f"Gemini's metadata:\n {response.usage_metadata}")
    logging.info("Answer generated!")

    thinking_content = ""
    answer = ""

    for part in response.candidates[0].content.parts:
        if not part.text:
            continue
        elif part.thought:
            if not thinking_content:
                print("Thoughts summary:")
            print(part.text)
            thinking_content += part.text
        else:
            if not answer:
                print("Answer:")
            print(part.text)
            answer += part.text

    logging.info(f"Thinking summary:\n{thinking_content}")

    return answer


def run_deepseek(model_name, prompt, temperature, top_p):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=api_key,
    )

    completion = client.chat.completions.create(
        model=get_model_id(model_name, DEEPSEEK_MODELS),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        stream=False
    )

    logging.info("LLM Usage:")
    print(completion.usage)

    answer = completion.choices[0].message.content

    return answer


def run_openai(model_name, prompt, temperature, top_p):

    # NOTE: Since GPT-5, OpenAI does not let us change the temperature either!
    # If this is the case, we produce a warning and set it to 1.0 which is the
    # only allowed value because it's their default
    # (Written on 31/08/2025, at 19:38 CEST)

    if model_name == 'gpt-5':
        logging.warning(f"Model {model_name} only supports temperature 1.0! "
                        "Automatically switching it...")
        temperature = 1.0

    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(
        api_key=api_key
    )

    completion = client.chat.completions.create(
        model=get_model_id(model_name, OPENAI_MODELS),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        #top_p=top_p, # OpenAI does not support top-p with o1
        stream=False
    )

    logging.info("LLM Usage:")
    print(completion.usage)

    answer = completion.choices[0].message.content

    return answer
