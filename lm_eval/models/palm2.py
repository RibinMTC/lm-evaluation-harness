"""
This code has been adapted from https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/big-refactor/lm_eval/models/openai_completions.py for GPT-4 experiments
"""
import os
import time
from google.oauth2 import service_account
from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel
from lm_eval.base import LM
from typing import List, Tuple
from tqdm import tqdm
from lm_eval import utils


class Palm2CompletionsLM(LM):
    REQ_CHUNK_SIZE = 20

    def __init__(
            self,
            engine: str = "text-bison@001",
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
        credentials = service_account.Credentials.from_service_account_file('gcloud_service_account.json')
        # get environment variables
        project = os.environ.get('GCLOUD_PROJECT')
        aiplatform.init(project=project, location='us-central1', credentials=credentials, experiment='llm-master-thesis')

        self.engine = engine

    # @property
    # def eot_token_id(self):
    #     return self.end_of_text_token_id

    @property
    def max_length(self) -> int:
        # Note: the OpenAI API supports up to 2049 tokens, with the first token being the first input token
        return 2048

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

    # def tok_encode(self, string: str) -> List[int]:
    #     return self.tokenizer.encode(string)
    #
    # def tok_decode(self, tokens: List[int]) -> str:
    #     return self.tokenizer.decode(tokens)

    def greedy_until(self, requests) -> List[str]:
        if not requests:
            return []

        parameters = {  # initialize to default values
            'max_output_tokens': 256,
            'temperature': 0.0,
            'top_p': 1.0,
            'top_k': 40,
            'safety_settings': [
                {
                    "category": "HARM_CATEGORY_UNSPECIFIED",
                    "threshold": "BLOCK_NONE"
                }, {
                    "category": "HARM_CATEGORY_DEROGATORY",
                    "threshold": "BLOCK_NONE"
                }, {
                    "category": "HARM_CATEGORY_TOXICITY",
                    "threshold": "BLOCK_NONE"
                }, {
                    "category": "HARM_CATEGORY_VIOLENCE",
                    "threshold": "BLOCK_NONE"
                }, {
                    "category": "HARM_CATEGORY_SEXUAL",
                    "threshold": "BLOCK_NONE"
                }, {
                    "category": "HARM_CATEGORY_MEDICAL",
                    "threshold": "BLOCK_NONE"
                }, {
                    "category": "HARM_CATEGORY_DANGEROUS",
                    "threshold": "BLOCK_NONE"
                }
            ]
        }

        model = TextGenerationModel.from_pretrained(self.engine)

        results = []
        for request in tqdm(requests):
            # messages = [{"role": "user", "content": request[0]}]

            # TODO: temperature parameter -> is it accessible here?
            response = model.predict(
                request[0],
                **parameters,
            )

            prediction = response.text
            results.append(prediction)
        return results

    def _encode_pair(
            self, context: str, continuation: str
    ) -> Tuple[List[int], List[int]]:
        # n_spaces = len(context) - len(context.rstrip())
        # if n_spaces > 0:
        #     continuation = context[-n_spaces:] + continuation
        #     context = context[:-n_spaces]
        # whole_enc = self.tok_encode(context + continuation)
        # context_enc = self.tok_encode(context)
        # context_enc_len = len(context_enc)
        # continuation_enc = whole_enc[context_enc_len:]
        # return context_enc, continuation_enc
        raise NotImplementedError()

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        # new_reqs = []
        # for context, continuation in [req.args for req in requests]:
        #     if context == "":
        #         # end of text as context
        #         context_enc, continuation_enc = [self.eot_token_id], self.tok_encode(
        #             continuation
        #         )
        #     else:
        #         context_enc, continuation_enc = self._encode_pair(context, continuation)
        #
        #     new_reqs.append(((context, continuation), context_enc, continuation_enc))
        #
        # return self._loglikelihood_tokens(new_reqs)
        raise NotImplementedError()

    def _loglikelihood_tokens(
            self, requests, disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        # res = []
        #
        # def _collate(x):
        #     # this doesn't efficiently handle last-token differences yet, but those are kinda annoying because
        #     # it's not guaranteed that the 100 or so logprobs we get to see actually contain all the continuations
        #     # we care about, and so we need some kind of backup for when it isn't
        #     toks = x[1] + x[2]
        #     return -len(toks), tuple(toks)
        #
        # re_ord = utils.Reorderer(requests, _collate)
        #
        # for chunk in tqdm(
        #         list(utils.chunks(re_ord.get_reordered(), self.REQ_CHUNK_SIZE)),
        #         disable=disable_tqdm,
        # ):
        #     inps = []
        #     ctxlens = []
        #     for cache_key, context_enc, continuation_enc in chunk:
        #         # max_length+1 because the API takes up to 2049 tokens, including the first context token
        #         inp = (context_enc + continuation_enc)[-(self.max_length + 1):]
        #         # TODO: the logic is much simpler if we just look at the length of continuation tokens
        #         ctxlen = len(context_enc) - max(
        #             0, len(context_enc) + len(continuation_enc) - (self.max_length + 1)
        #         )
        #
        #         inps.append(inp)
        #         ctxlens.append(ctxlen)
        #
        #     response = oa_completion(
        #         engine=self.engine,
        #         prompt=inps,
        #         echo=True,
        #         max_tokens=0,
        #         temperature=0.0,
        #         logprobs=10,
        #     )
        #
        #     for resp, ctxlen, (cache_key, context_enc, continuation_enc) in zip(
        #             response.choices, ctxlens, chunk
        #     ):
        #         answer = get_result(resp, ctxlen)
        #
        #         res.append(answer)
        #
        #         # partial caching
        #         if cache_key is not None:
        #             self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        # return re_ord.get_original(res)
        raise NotImplementedError()

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()

    def loglikelihood_rolling(self, requests) -> List[float]:
        # loglikelihoods = []
        #
        # for (string,) in tqdm([req.args for req in requests]):
        #     rolling_token_windows = list(
        #         map(
        #             utils.make_disjoint_window,
        #             utils.get_rolling_token_windows(
        #                 token_list=self.tok_encode(string),
        #                 prefix_token=self.eot_token_id,
        #                 max_seq_len=self.max_length,
        #                 context_len=1,
        #             ),
        #         )
        #     )
        #
        #     # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
        #     rolling_token_windows = [(None,) + x for x in rolling_token_windows]
        #
        #     string_nll = self._loglikelihood_tokens(
        #         rolling_token_windows,
        #         disable_tqdm=True,
        #     )
        #
        #     # discard is_greedy
        #     string_nll = [x[0] for x in string_nll]
        #
        #     string_nll = sum(string_nll)
        #     loglikelihoods.append(string_nll)
        # return loglikelihoods
        raise NotImplementedError()
