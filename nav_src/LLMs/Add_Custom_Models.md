## Add Custom LLMs for NavGPT

## Contents

- [Set up built-in integrations with LLM providers](#set-up-built-in-integrations-with-llm-providers)
- [Set up local model inference](#set-up-local-model-inference)
  - [Step 1: Set up the model environment](#step-1-set-up-the-model-environment)
  - [Step 2: Set up the inference pipeline](#step-2-set-up-the-inference-pipeline)
  - [Step 3: Register the custom LLM](#step-3-register-the-custom-llm)
  - [Step 4: Run NavGPT with the custom LLM](#step-4-run-navgpt-with-the-custom-llm)

## Set up built-in integrations with LLM providers

The `Langchain` package has integrated various cloud services which provide LLMs inference APIs ([OpenAI](https://openai.com/), [Cohere](https://cohere.ai/), [Hugging Face](https://huggingface.co/), etc). You can use these services directly by setting up the API keys.

You can also check out the [Langchain Integrations Documentations](https://python.langchain.com/docs/integrations/llms/) for more information.

## Set up local model inference

One possible way to set up local inference is through [Hugging Face Loacal Pipelines](https://python.langchain.com/docs/integrations/llms/huggingface_pipelines) in Langchain.

However, to maximize the degree of freedom of running local inference or setting up your custum LLMs, we recommend you to set up your own inference pipeline. We provide an example of `nav_src/LLMs/Langchain_llama.py` to show how to set up a local inference pipeline.

You can check out the [Langchain Custom LLM](https://python.langchain.com/docs/modules/model_io/models/llms/custom_llm) for more information.

We will use Llama-2 as an example to show how to set up a local inference pipeline.

### Step 1: Set up the model environment
Add the Llama-2 repo as a submodule under `nav_src/LLMs/`:
```bash
cd nav_src/LLMs
git submodule add https://github.com/facebookresearch/llama.git
```
Because we have already set up the `nav_src/LLMs/llama` as a submodule, you can skip the previous step, initialize and clone the submodule by:
```bash
git submodule update --init --recursive
```

Download the [Llama-2 weights](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) accroding to the [instructions](https://github.com/facebookresearch/llama) and set up the Llama-2 environment:
```bash
cd llama
pip install -e .
```

### Step 2: Set up the inference pipeline
Create your own LLM class `Custom_model` under `nav_src/LLMs/Langchain_model.py`:

There is only one required `_call` function that a custom LLM needs to implement, for example:
```python
def _call(
    self,
    prompt: str,
    stop: Optional[List[str]] = None,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> str:

    if stop is not None:
        raise ValueError("stop kwargs are not permitted.")

    result = self.model.generate(
        prompt,
        max_length=self.max_length,
        num_beams=self.num_beams,
        temperature=self.temperature,
        top_k=self.top_k,
        top_p=self.top_p,
        repetition_penalty=self.repetition_penalty,
        do_sample=self.do_sample,
        num_return_sequences=self.num_return_sequences,
        **kwargs,
    )
    return result
```

An optional `_identifying_params` property can be rewrited to help with printing of this class. Should return a dictionary.
```python
@property
def _identifying_params(self) -> Mapping[str, Any]:
    """Get the identifying parameters."""
    return {
        "model_name": self.model_name,
        "max_length": self.max_length,
        "num_beams": self.num_beams,
        "temperature": self.temperature,
        "top_k": self.top_k,
        "top_p": self.top_p,
        "repetition_penalty": self.repetition_penalty,
        "do_sample": self.do_sample,
        "num_return_sequences": self.num_return_sequences,
    }
```

If your custom LLM needs to be initialized with some parameters, you can write your own `from_config` or `from_model_id` classmethod. Check out the example in `nav_src/LLMs/Langchain_llama.py` for more information.

Here is an example of running our custom Llama-2 locally as a LLMChain in Langchain:
```python
>>> from langchain import PromptTemplate, LLMChain
>>> from nav_src.LLMs.Langchain_llama import Custom_Llama

>>> ckpt_dir = "LLMs/llama/llama-2-13b"
>>> tokenizer_path = "LLMs/llama/tokenizer.model"

>>> llm = Custom_Llama.from_model_id(
        temperature=0.75,
        ckpt_dir = ckpt_dir,
        tokenizer_path = tokenizer_path,
        max_seq_len = 4000,
        max_gen_len = 800,
        max_batch_size = 4,
    )

>>> template = """Question: {question}\nAnswer: Let's think step by step."""
>>> prompt = PromptTemplate(template=template, input_variables=["question"])

>>> llm_chain = LLMChain(prompt=prompt, llm=llm)

>>> question = "What is electroencephalography?"
>>> print(llm_chain.run(question))

"Sure, I'd be happy to help! Here's a step-by-step explanation of what electroencephalography (EEG) is:
1. Electroencephalography (EEG) is a non-invasive neuroimaging technique that measures the electrical activity of the brain.
2. The brain is made up of billions of neurons, which communicate with each other through electrical signals. EEG recordings measure these electrical signals, allowing researchers and clinicians to study the brain's activity.
3. To record EEG data, electrodes are placed on the scalp, usually in a specific pattern such as the International 10-20 system. These electrodes detect the electrical activity of the brain and transmit it to a computer for analysis.
4. The EEG signal is composed of different frequency bands, including alpha, beta, gamma, and theta waves. Each frequency band is associated with different cognitive processes, such as attention, relaxation, or memory.
5. EEG can be used to diagnose and monitor a variety of neurological conditions, such as epilepsy, sleep disorders, and stroke. It can also be used to assess brain function in patients with traumatic brain injury, coma, or vegetative state.
6. In addition to diagnostic applications, EEG is also used in research studies to investigate the neural mechanisms underlying various cognitive processes, such as language processing, memory formation, and decision-making.
7. EEG has several advantages over other neuroimaging techniques, such as functional magnetic resonance imaging (fMRI) or positron emission tomography (PET). For example, EEG is relatively inexpensive, portable, and can be performed in a clinical setting or at home. Additionally, EEG provides high temporal resolution, allowing researchers to study the dynamics of brain activity in real-time.
8. Overall, EEG is a valuable tool for understanding the workings of the human brain, diagnosing neurological conditions, and monitoring brain health. Its non-invasive nature and high temporal resolution make it an important technique in neuroscience research and clinical practice."
```

### Step 3: Register the custom LLM
In `nav_src/agent.py`, register the custom LLM by adding the following code after `line 176`:
```python
elif config.llm_model_name == 'your_custom_llm':
    from LLMs.Langchain_model import Custom_model
    self.llm = Custom_model.from_config(
        config = config,
    )
```

### Step 4: Run NavGPT with the custom LLM
Now you can run NavGPT with your custom LLM:
```bash
cd nav_src
python NavGPT.py --llm_model_name your_custom_llm \
    --output_dir ../datasets/R2R/exprs/your_custom_llm-test
```