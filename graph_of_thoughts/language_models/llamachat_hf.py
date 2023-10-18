# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Ales Kubicek

import os
import torch
from typing import List, Dict, Union
from .abstract_language_model import AbstractLanguageModel


class Llama2HF(AbstractLanguageModel):
    """
    An interface to use LLaMA 2 models through the HuggingFace library.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "llama7b-hf", cache: bool = False
    ) -> None:
        """
        Initialize an instance of the Llama2HF class with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to an empty string.
        :type config_path: str
        :param model_name: Specifies the name of the LLaMA model variant. Defaults to "llama7b-hf".
                           Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        # Detailed id of the used model.
        self.model_id: str = self.config["model_id"]
        # Costs for 1000 tokens.
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        # The temperature is defined as the randomness of the model's output.
        self.temperature: float = self.config["temperature"]
        # Top K sampling.
        self.top_k: int = self.config["top_k"]
        # The maximum number of tokens to generate in the chat completion.
        self.max_tokens: int = self.config["max_tokens"]

        # Important: must be done before importing transformers
        os.environ["TRANSFORMERS_CACHE"] = self.config["cache_dir"]
        import transformers

        hf_model_id = f"meta-llama/{self.model_id}"
        model_config = transformers.AutoConfig.from_pretrained(hf_model_id)
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_id)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.model.eval()
        torch.no_grad()

        self.generate_text = transformers.pipeline(
            model=self.model, tokenizer=self.tokenizer, task="text-generation"
        )

    def query(self, query: str, num_responses: int = 1) -> List[Dict]:
        """
        Query the LLaMA 2 model for responses.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the LLaMA 2 model.
        :rtype: List[Dict]
        """
        if self.cache and query in self.respone_cache:
            return self.respone_cache[query]
        sequences = []
        query = f"<s><<SYS>>You are a helpful assistant. Always follow the intstructions precisely and output the response exactly in the requested format.<</SYS>>\n\n[INST] {query} [/INST]"
        for _ in range(num_responses):
            sequences.extend(
                self.generate_text(
                    query,
                    do_sample=True,
                    top_k=self.top_k,
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_tokens,
                )
            )
        response = [
            {"generated_text": sequence["generated_text"][len(query) :].strip()}
            for sequence in sequences
        ]
        if self.cache:
            self.respone_cache[query] = response
        return response

    def get_response_texts(self, query_responses: List[Dict]) -> List[str]:
        """
        Extract the response texts from the query response.

        :param query_responses: The response list of dictionaries generated from the `query` method.
        :type query_responses: List[Dict]
        :return: List of response strings.
        :rtype: List[str]
        """
        return [query_response["generated_text"] for query_response in query_responses]
