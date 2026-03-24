# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach
# contributor: octopus (MiniMax integration)

import backoff
import os
import random
import time
from typing import List, Dict, Union
from openai import OpenAI, OpenAIError
from openai.types.chat.chat_completion import ChatCompletion

from .abstract_language_model import AbstractLanguageModel


class MiniMax(AbstractLanguageModel):
    """
    The MiniMax class handles interactions with the MiniMax models using the
    OpenAI-compatible API at https://api.minimax.io/v1.

    Inherits from the AbstractLanguageModel and implements its abstract methods.
    """

    def __init__(
        self,
        config_path: str = "",
        model_name: str = "minimax",
        cache: bool = False,
    ) -> None:
        """
        Initialize the MiniMax instance with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the model, default is 'minimax'. Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        # The model_id is the id of the model that is used, e.g. MiniMax-M2.7
        self.model_id: str = self.config["model_id"]
        # The prompt_token_cost and response_token_cost are the costs for 1000 prompt tokens and 1000 response tokens respectively.
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        # The temperature of a model is defined as the randomness of the model's output.
        # MiniMax temperature range is [0, 1].
        self.temperature: float = min(
            max(self.config["temperature"], 0.0), 1.0
        )
        # The maximum number of tokens to generate in the chat completion.
        self.max_tokens: int = self.config["max_tokens"]
        # The stop sequence is a sequence of tokens that the model will stop generating at.
        self.stop: Union[str, List[str]] = self.config["stop"]
        # MiniMax API key
        self.api_key: str = os.getenv("MINIMAX_API_KEY", self.config["api_key"])
        if self.api_key == "":
            raise ValueError("MINIMAX_API_KEY is not set")
        # MiniMax API base URL
        self.base_url: str = self.config.get(
            "base_url", "https://api.minimax.io/v1"
        )
        # Initialize the OpenAI-compatible client for MiniMax
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def query(
        self, query: str, num_responses: int = 1
    ) -> Union[List[ChatCompletion], ChatCompletion]:
        """
        Query the MiniMax model for responses.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the MiniMax model.
        :rtype: Union[List[ChatCompletion], ChatCompletion]
        """
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        if num_responses == 1:
            response = self.chat(
                [{"role": "user", "content": query}], num_responses
            )
        else:
            # MiniMax does not support the n parameter for multiple responses
            # in a single call, so we make individual calls.
            response = []
            next_try = num_responses
            total_num_attempts = num_responses
            while num_responses > 0 and total_num_attempts > 0:
                try:
                    assert next_try > 0
                    for _ in range(next_try):
                        res = self.chat(
                            [{"role": "user", "content": query}], 1
                        )
                        response.append(res)
                    num_responses -= next_try
                    next_try = min(num_responses, next_try)
                except Exception as e:
                    next_try = (next_try + 1) // 2
                    self.logger.warning(
                        f"Error in MiniMax: {e}, trying again with {next_try} samples"
                    )
                    time.sleep(random.randint(1, 3))
                    total_num_attempts -= 1

        if self.cache:
            self.response_cache[query] = response
        return response

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    def chat(
        self, messages: List[Dict], num_responses: int = 1
    ) -> ChatCompletion:
        """
        Send chat messages to the MiniMax model and retrieve the model's response.
        Implements backoff on OpenAI-compatible error.

        :param messages: A list of message dictionaries for the chat.
        :type messages: List[Dict]
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: The MiniMax model's response.
        :rtype: ChatCompletion
        """
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=self.stop,
        )

        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        prompt_tokens_k = float(self.prompt_tokens) / 1000.0
        completion_tokens_k = float(self.completion_tokens) / 1000.0
        self.cost = (
            self.prompt_token_cost * prompt_tokens_k
            + self.response_token_cost * completion_tokens_k
        )
        self.logger.info(
            f"This is the response from MiniMax: {response}"
            f"\nThis is the cost of the response: {self.cost}"
        )
        return response

    def get_response_texts(
        self, query_response: Union[List[ChatCompletion], ChatCompletion]
    ) -> List[str]:
        """
        Extract the response texts from the query response.

        :param query_response: The response dictionary (or list of dictionaries) from the MiniMax model.
        :type query_response: Union[List[ChatCompletion], ChatCompletion]
        :return: List of response strings.
        :rtype: List[str]
        """
        if not isinstance(query_response, List):
            query_response = [query_response]
        return [
            choice.message.content
            for response in query_response
            for choice in response.choices
        ]
