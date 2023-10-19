from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from LLMs.llama.llama import Llama

class Custom_Llama(LLM):
    model: Any  #: :meta private:

    """Key word arguments passed to the model."""
    ckpt_dir: str
    tokenizer_path: str
    temperature: float = 0.6
    top_p: float = 0.9
    max_seq_len: int = 128
    max_gen_len: int = 64
    max_batch_size: int = 4

    @property
    def _llm_type(self) -> str:
        return "custom_llama"

    @classmethod
    def from_model_id(
        cls,
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 128,
        max_gen_len: int = 64,
        max_batch_size: int = 4,
        **kwargs: Any,
    ) -> LLM:
        """Construct the pipeline object from model_id and task."""

        model = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )

        return cls(
            model = model,
            ckpt_dir = ckpt_dir,
            tokenizer_path = tokenizer_path,
            # set as default
            temperature = 0.6,
            top_p = top_p,
            max_seq_len = max_seq_len,
            max_gen_len = max_gen_len,
            max_batch_size = max_batch_size,
            **kwargs,
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")

        result = self.model.text_completion(
            [prompt],
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return result[0]["generation"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "ckpt_dir": self.ckpt_dir,
            "tokenizer_path": self.tokenizer_path,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_seq_len": self.max_seq_len,
            "max_gen_len": self.max_gen_len,
            "max_batch_size": self.max_batch_size,
            }