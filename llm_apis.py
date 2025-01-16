import collections
import json
import logging
import os
import time
from typing import Any, Dict, List, Literal, NamedTuple
import warnings

import requests
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether

Reponse = collections.namedtuple("Response", ["content"])


class HUITLLM:
    """
    Custom chat model for a HUIT AWS Bendrock endpoint.
    **Only for internal use at Harvard.
    """

    def __init__(
        self,
        model: str = "mistral.mistral-large-2407-v1:0",
        max_attempts: int = 3,
        wait_time_between_attempts: int = 15,
    ):
        valid = ["mistral", "meta"]
        assert any([model.startswith(v) for v in valid]), f"Invalid model: {model}"
        metadata = {}
        metadata["endpoint_url"] = "https://go.apis.huit.harvard.edu/ais-bedrock-llm/v1"
        metadata["api_key"] = os.getenv("HUIT_AI_API_KEY")
        self.model = model
        self.metadata = metadata
        self.max_attempts = max_attempts
        self.wait_time_between_attempts = wait_time_between_attempts

    @property
    def max_tokens_key(self) -> str:
        if self.model.startswith("mistral"):
            return "max_tokens"
        elif self.model.startswith("meta"):
            return "max_gen_len"
        else:
            raise ValueError(f"Invalid model: {self.model}")

    def invoke(
        self,
        messages: List[Dict[Literal["role", "content"], str]],
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> NamedTuple:
        # 1. Convert from LangChain messages -> the custom format
        aws_style_messages = []
        for msg in messages:
            aws_style_messages.append(
                {
                    "role": msg["role"],
                    "content": [{"type": "text", "text": str(msg["content"])}],
                }
            )

        # 2. Construct the payload
        payload = json.dumps(
            {
                "modelId": self.model,
                "contentType": "application/json",
                "accept": "application/json",
                "body": {
                    "messages": aws_style_messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    self.max_tokens_key: max_tokens,
                },
            }
        )

        # 3. Send the request
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.metadata["api_key"],
        }

        attempts = 0
        while attempts < self.max_attempts:
            try:
                response = requests.post(
                    self.metadata["endpoint_url"],
                    headers=headers,
                    data=payload,
                    # timeout=60,
                )

                response.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                warnings.warn(f"Attempt {attempts} failed: {e}")
                attempts += 1
                time.sleep(self.wait_time_between_attempts)

        # 4. Parse the response
        #    Adjust based on how your custom endpoint returns the data
        #    For example, maybe the response has a field "generated_text":
        # content = result_json.get("generated_text", "")
        result_json = response.json()
        content = result_json["choices"][0]["message"]["content"]

        return Reponse(content=content)


class HUITOpenAI:
    """
    Custom chat model for a HUIT OpenAI endpoint.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_attempts: int = 3,
        wait_time_between_attempts: int = 3,
    ):
        metadata = {}
        metadata["endpoint_url"] = (
            "https://go.apis.huit.harvard.edu/ais-openai-direct/v1/chat/completions"
        )
        metadata["api_key"] = os.getenv("HUIT_AI_API_KEY")
        self.model = model.replace("-huit", "")
        self.metadata = metadata
        self.max_attempts = max_attempts
        self.wait_time_between_attempts = wait_time_between_attempts

    def invoke(
        self,
        messages: List[Dict[Literal["role", "content"], str]],
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> NamedTuple:
        # 1. Construct the payload
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )

        headers = {
            "Content-Type": "application/json",
            "api-key": self.metadata["api_key"],
        }

        # 2. Send the request
        attempts = 0
        while attempts < self.max_attempts:
            try:
                response = requests.post(
                    self.metadata["endpoint_url"],
                    headers=headers,
                    data=payload,
                )

                response.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                warnings.warn(f"Attempt {attempts} failed: {e}")
                attempts += 1
                time.sleep(self.wait_time_between_attempts)

        if attempts >= self.max_attempts:
            raise RuntimeError("Failed to get a response from the endpoint.")

        # 3. Parse the response
        result_json = response.json()
        content = result_json["choices"][0]["message"]["content"]

        return Reponse(content=content)


ModelAPIDict = {
    "google/gemma-2b-it": ChatTogether,
    "meta-llama/Llama-3.2-3B-Instruct-Turbo": ChatTogether,
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": ChatTogether,
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": ChatTogether,
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": ChatTogether,
    "meta.llama3-1-8b-instruct-v1:0": HUITLLM,
    "meta.llama3-1-70b-instruct-v1:0": HUITLLM,
    "meta.llama3-2-3b-instruct-v1:0": HUITLLM,
    "meta.llama3-3-70b-instruct-v1:0": HUITLLM,
    "gpt-4o-mini-huit": HUITOpenAI,
    "gpt-4o-mini": ChatOpenAI,
}

ValidModels = Literal[
    "google/gemma-2b-it",
    "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta.llama3-1-8b-instruct-v1:0",
    "meta.llama3-1-70b-instruct-v1:0",
    "meta.llama3-2-3b-instruct-v1:0",
    "meta.llama3-3-70b-instruct-v1:0",
    "gpt-4o-mini-huit",
    "gpt-4o-mini",
]


def invoke_with_retries(
    model: ValidModels,
    messages: List[Dict[Literal["role", "content"], str]],
    *args,
    max_attempts: int = 3,
    wait_time_between_attempts: int = 60,
    **kwargs,
):
    attempts = 0
    while attempts < max_attempts:
        try:
            result = model.invoke(messages, *args, **kwargs)
            return result
        except Exception as e:
            logging.error(f"Attempt {attempts} failed: {e}")
            attempts += 1
            time.sleep(wait_time_between_attempts)


def get_llm_api(model: ValidModels) -> Any:
    return ModelAPIDict[model](model=model)


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "Be angry! Answer angry to every message."},
        {"role": "user", "content": "What's your mood?"},
    ]

    # llm = HUITLLM("mistral.mistral-large-2407-v1:0")
    # result = llm.invoke(messages, max_tokens=10)
    # print(result.content)

    llm = HUITOpenAI("gpt-4o-mini")
    result = llm.invoke(messages, max_tokens=10)
    print(result.content)
