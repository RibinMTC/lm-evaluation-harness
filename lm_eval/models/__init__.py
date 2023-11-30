from . import gpt2
from . import gpt3
from . import huggingface
from . import textsynth
from . import dummy
from . import chat_gpt_model
from . import cpu_ctransformers

MODEL_REGISTRY = {
    "hf": gpt2.HFLM,
    "hf-causal": gpt2.HFLM,
    "hf-causal-experimental": huggingface.AutoCausalLM,
    "hf-seq2seq": huggingface.AutoSeq2SeqLM,
    "gpt2": gpt2.GPT2LM,
    "gpt3": gpt3.GPT3LM,
    "gpt3.5": chat_gpt_model.OpenaiCompletionsLM,
    "gpt4": chat_gpt_model.OpenaiCompletionsLM,
    "textsynth": textsynth.TextSynthLM,
    "dummy": dummy.DummyLM,
    "ctransformers-casual": cpu_ctransformers.CTransformersAutoLM
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
