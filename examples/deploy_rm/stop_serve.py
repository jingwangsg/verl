from ray import serve
import ray

ray.init(address="auto")
serve.shutdown()