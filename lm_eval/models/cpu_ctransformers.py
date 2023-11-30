import torch

from typing import List, Optional, Tuple, Union

from ctransformers import AutoModelForCausalLM

from lm_eval.base import LM


class CTransformersAutoLM(LM):

    def __init__(
            self,
            pretrained: str,
            subfolder: Optional[str] = None,
            model_type: Optional[str] = "mistral",
            gpu_layers: Optional[int] = 0,
            max_gen_toks: Optional[int] = 256,
            max_length: Optional[int] = None,
            device: Optional[Union[int, str]] = "cpu",
            temperature: float = 0.0,
            **kwargs
    ):

        super().__init__()

        self._max_gen_toks = max_gen_toks
        self._max_length = max_length

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id=pretrained, model_file=subfolder,
            model_type=model_type, gpu_layers=gpu_layers, context_length=max_length)

        self._device = device
        self.temperature = temperature

    @property
    def eot_token(self) -> str:
        return self.model.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.model.eos_token_id

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def device(self) -> Union[int, str, torch.device]:
        return self._device

    def tok_encode(self, string: str) -> List[int]:
        return self.model.tokenize(string)

    def tok_decode(self, tokens: List[int]) -> str:
        return self.model.detokenize(tokens=tokens)

    def greedy_until(
            self, requests: List[Tuple[str, Union[List[str], str]]]
    ) -> List[str]:

        results = []

        for request in requests:
            context = request[0]
            request_args = request[1]
            stop = request_args.get("until", None)
            stop_sequences = stop if isinstance(stop, list) else [stop]
            max_generation_length = request_args.get("max_length", None)

            assert (
                    isinstance(max_generation_length, int) or max_generation_length is None
            )
            assert isinstance(stop_sequences, list) or stop_sequences is None

            if max_generation_length is None:
                max_tokens = self.max_gen_toks
            else:
                max_tokens = max_generation_length

            response = self.model(
                prompt=context,
                temperature=self.temperature,
                max_new_tokens=max_tokens
            )

            print(f"Final output: {response}")
            results.append(response)

        return results

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
