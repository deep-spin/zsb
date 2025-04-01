from litellm import completion
from tenacity import Retrying, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from vllm import LLM, SamplingParams


class Model:
    def __init__(
        self,
        model_args: dict = {"proper_model_args": {}, "sampling_params": {}},
        **kwargs,
    ) -> None:
        pass

    def batch_generate(self, insts: list[str], imgs: list[str] = None) -> list[str]:
        if imgs is None:
            imgs = [None] * len(insts)
        generations = [
            self.generate(inst, img)
            for inst, img in tqdm(
                zip(insts, imgs), desc="Generating...", total=len(insts)
            )
        ]
        print(f"Example generation:\n\n{generations[-1]}")
        return generations

    def generate(self, inst: str, image_str: str) -> str:
        pass

    @staticmethod
    def generate_with_retries(
        retry_function,
        model_args,
        retry_max_attempts=1,
        retry_multiplier=1,
        retry_max_interval=10,
        retry_min_interval=4,
    ):
        retryer = Retrying(
            stop=stop_after_attempt(retry_max_attempts),
            wait=wait_random_exponential(
                multiplier=retry_multiplier,
                max=retry_max_interval,
                min=retry_min_interval,
            ),
            reraise=True,
        )
        return retryer(retry_function, **model_args)

    @staticmethod
    def convert_string_to_message(
        inst: str, system_prompt: str = None, image_str: str = None
    ) -> dict:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        if image_str is None:
            user_content = inst
        else:
            user_content = [
                {"type": "text", "text": inst},
                {"type": "image_url", "image_url": {"url": image_str}},
            ]
        messages.append({"role": "user", "content": user_content})

        return messages

    @staticmethod
    def model_type():
        pass


class VLLM(Model):
    def __init__(
        self,
        model_args: dict = {
            "proper_model_args": {},
            "sampling_params": {"temperature": 0, "max_tokens": 8192},
        },
        **kwargs,
    ) -> None:
        super().__init__(model_args, **kwargs)
        assert (
            "model" in model_args["proper_model_args"]
        ), "Model name must be provided for vllm models."

        model_args["proper_model_args"]["trust_remote_code"] = True
        sampling_args = model_args["sampling_params"]
        self.model = LLM(**model_args["proper_model_args"])
        self.sampling_params = SamplingParams(**sampling_args)

        self.system_prompt = model_args.get("system_prompt", None)

    def batch_generate(
        self, insts: list[str], image_strs: list[str] = None
    ) -> list[str]:
        if image_strs is None:
            image_strs = [None] * len(insts)
        messages = [
            self.convert_string_to_message(inst, self.system_prompt, image_str)
            for inst, image_str in zip(insts, image_strs)
        ]
        model_output = self.model.chat(messages, self.sampling_params, use_tqdm=True)
        generations = [output.outputs[0].text for output in model_output]
        assert len(generations) == len(insts), "Number of outputs must match inputs."
        print(f"Example generation:\n\n{generations[-1]}")
        return generations

    @staticmethod
    def model_type():
        return "vllm"


class LiteLLM(Model):
    def __init__(
        self,
        model_args={
            "proper_model_args": {},
            "sampling_params": {"temperature": 0, "max_tokens": 4096},
        },
        **kwargs,
    ) -> None:
        super().__init__(model_args, **kwargs)
        model = model_args["proper_model_args"].get(
            "model", "claude-3-5-sonnet-20241022"
        )
        temperature = model_args["sampling_params"].get("temperature", 0)
        max_tokens = model_args["sampling_params"].get("max_tokens", 8192)
        self.model_args = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        self.system_prompt = model_args.get("system_prompt", None)

    def generate(self, inst: str, image_str: str = None) -> str:
        messages = self.convert_string_to_message(inst, self.system_prompt, image_str)
        self.model_args["messages"] = messages
        response = completion(**self.model_args, max_retries=200)
        model_output = response.choices[0].message.content
        return model_output

    @staticmethod
    def model_type():
        return "litellm"


def instantiate_model(model_type: str, model_args: dict) -> Model:
    available_models = {
        VLLM.model_type(): VLLM,
        LiteLLM.model_type(): LiteLLM,
    }
    try:
        instantiated_model = available_models[model_type](model_args=model_args)
    except KeyError:
        raise ValueError(
            f"Model {model_type} not available. Choose from {available_models.keys()}."
        )

    return instantiated_model
