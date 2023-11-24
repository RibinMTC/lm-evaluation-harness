"""
This code has been adapted from https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/big-refactor/lm_eval/models/openai_completions.py for GPT-4 experiments
"""
import os
import time
import openai
from lm_eval.base import LM
from typing import List, Tuple
from tqdm import tqdm
from lm_eval import utils
from dotenv import load_dotenv


# def get_result(response: dict, ctxlen: int) -> Tuple[float, bool]:
#     """Process results from OpenAI API response.
#
#     :param response: dict
#         OpenAI API Response
#     :param ctxlen: int
#         Length of context (so we can slice them away and only keep the predictions)
#     :return:
#         continuation_logprobs: np.array
#             Log probabilities of continuation tokens
#         is_greedy: bool
#             whether argmax matches given continuation exactly
#     """
#     is_greedy = True
#     logprobs = response["logprobs"]["token_logprobs"]
#     continuation_logprobs = sum(logprobs[ctxlen:])
#
#     for i in range(ctxlen, len(response["logprobs"]["tokens"])):
#         token = response["logprobs"]["tokens"][i]
#         top_tokens = response["logprobs"]["top_logprobs"][i]
#         top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
#         if top_token != token:
#             is_greedy = False
#             break
#
#     return continuation_logprobs, is_greedy


def oa_completion(**kwargs):
    """Query OpenAI API for completion.

    Retry with back-off until they respond
    """
    try:
        import openai, tiktoken  # noqa: E401
    except ModuleNotFoundError:
        raise Exception(
            "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. \
please install these via `pip install lm-eval[openai]` or `pip install -e .[openai]`",
        )

    backoff_time = 3
    while True:
        try:
            return openai.ChatCompletion.create(**kwargs)
        except openai.error.OpenAIError:
            import traceback

            traceback.print_exc()
            time.sleep(backoff_time)
            backoff_time *= 1.5


class OpenaiCompletionsLM(LM):
    REQ_CHUNK_SIZE = 20

    def __init__(
            self,
            engine: str = "gpt-3.5-turbo",
            truncate: bool = False,
            batch_size: int = 1,
    ) -> None:
        """

        :param engine: str
            OpenAI API engine (e.g. davinci)
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()
        try:
            import openai, tiktoken  # noqa: E401
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. \
    please install these via `pip install lm-eval[openai]` or `pip install -e .[openai]`",
            )
        self.engine = engine
        self.tokenizer = tiktoken.encoding_for_model(self.engine)
        self.vocab_size = self.tokenizer.n_vocab
        self.truncate = truncate
        self.end_of_text_token_id = self.tokenizer.eot_token

        # Read from environment variable OPENAI_API_SECRET_KEY
        load_dotenv()
        openai.api_key = os.environ["OPENAI_API_SECRET_KEY"]

    @property
    def eot_token_id(self):
        return self.end_of_text_token_id

    @property
    def max_length(self) -> int:
        # Note: the OpenAI API supports up to 2049 tokens, with the first token being the first input token
        return 4096

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def greedy_until(self, requests) -> List[str]:
        if not requests:
            return []

        results = []
        for request in tqdm(requests):
            request_suffix = request[1].get("request_suffix", "")
            messages = [{"role": "user", "content": request[0] + request_suffix}]
            max_generation_length = request[1].get("max_length", None)
            # TODO: temperature parameter -> is it accessible here?
            response = oa_completion(
                model=self.engine,
                messages=messages,
                max_tokens=max_generation_length,
                temperature=0.0,
            )

            # prepare the response
            if len(response.choices) == 0:
                prediction = ""
            else:
                prediction = response.choices[0].message.content
            results.append(prediction)
        return results

    def _encode_pair(
            self, context: str, continuation: str
    ) -> Tuple[List[int], List[int]]:
        raise NotImplementedError()

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        raise NotImplementedError()

    def _loglikelihood_tokens(
            self, requests, disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError()

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()

    def loglikelihood_rolling(self, requests) -> List[float]:
        raise NotImplementedError()