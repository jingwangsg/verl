import uuid

import ray
from ray import serve

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.inputs import TextPrompt
from argparse import ArgumentParser
import ray._private.worker as _ray_worker

parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
parser.add_argument("--tensor_parallel_size", type=int, default=1)
parser.add_argument("--num_replicas", type=int, default=1)
parser.add_argument("--max_tokens", type=int, default=2048)
parser.add_argument("--name", type=str, default="rm_vllm")
args = parser.parse_args()


@serve.deployment(
    ray_actor_options={"num_gpus": args.tensor_parallel_size},
    num_replicas=args.num_replicas,
)
class AsyncVLLMRayServer:
    def __init__(self):
        # vLLM 官方推荐：用 AsyncEngineArgs + AsyncLLMEngine.from_engine_args
        engine_args = AsyncEngineArgs(
            model=args.model_path,
            trust_remote_code=True,  # Qwen3 自定义结构，官方文档建议打开 [oai_citation:8‡Qwen](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html?utm_source=chatgpt.com)
            tensor_parallel_size=args.tensor_parallel_size,  # 单卡 7B，够用就写 1；多卡自己调
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.default_sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
        )

    async def generate(self, prompt: TextPrompt) -> str:
        """内部封装一次生成调用。"""
        request_id = str(uuid.uuid4())

        # vLLM AsyncLLMEngine 典型用法：engine.generate -> async generator，
        # async for 读到最后一个输出再取 text [oai_citation:9‡Medium](https://medium.com/%40crclq2018/explaining-the-source-code-behind-the-vllm-fast-inference-engine-91429f54d1f7?utm_source=chatgpt.com)
        results = self.engine.generate(
            prompt=prompt,
            sampling_params=self.default_sampling_params,
            request_id=request_id,
        )

        final_output = None
        async for output in results:
            final_output = output

        return final_output

app = AsyncVLLMRayServer.bind()

if __name__ == "__main__":
    ray.init(address="auto", log_to_driver=True)
    serve.run(app, name=args.name, route_prefix="/rm_vllm", blocking=True)
