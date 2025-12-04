# async_query_30.py
import asyncio
import ray
from ray import serve
import time
from vllm.inputs import TextPrompt
from debug.snapshot import Snapshot


async def infer_one(handle, prompt: str, idx: int):
    """对单条 prompt 发请求并返回结果。"""
    # 注意：generate.remote 返回的是 DeploymentResponse，可以直接 await
    resp = handle.generate.remote(TextPrompt(prompt=prompt))
    result = await resp
    print(f"[#{idx}] prompt: {prompt[:20]}... -> reply 前 40 字: {result[:40]}...")
    return result


async def main():

    # 拿到 rm_vllm 这个 Serve app 的 handle
    handle = serve.get_app_handle("rm_vllm")

    # 30 条差异比较大的 prompt
    prompts = [
        # 1–5 交易 / 财经
        "用不超过 200 字总结一下：如果未来 6 个月美联储延后降息、油价维持高位，对亚洲炼厂利润和成品油裂解价差可能有哪些影响？分点回答。",
        "我今天做了一笔 MOPJ 月差多头，但被 intraday 波动打掉止损。请从“入场逻辑、仓位管理、风险控制、复盘改进”四个角度帮我写一段冷静客观的复盘。",
        "写一份发给交易团队的“本周风险提示”模板，要求有：标题、三条核心风险点、两条操作建议，语气专业克制，避免空话。",
        "假设你在回答一个完全不懂期货的小白，用通俗类比解释什么是“月差交易”，并举一个和现实生活（比如预订机票/订酒店）相关的类比。",
        "写一小段 150 字以内的文字，主题是“当市场和你的观点相反时，如何保证执行力”，风格偏理性、冷静，不要鸡汤。",
        # 6–10 科研 / 写作
        "请把下面这句话改写成适合放在论文 introduction 开头的一句话，要求严肃、学术：“视频推理模型现在很会瞎编，我们想办法强迫它老老实实看证据。”",
        "Reviewer 说：“The proposed method seems incremental and lacks a strong ablation on component X.” 请帮我分析他背后的真实意图可能有哪些，并给出 3 种不同层次的回应策略大纲（从“最低成本应付”到“高质量彻底回应”）。",
        "用“三段式结构：现状问题 → 核心矛盾 → 研究切入点”写一段关于“长视频推理模型 evaluation 难在哪里”的分析，不超过 250 字。",
        "把“我们发现模型经常在没有看完整视频的情况下就开始乱猜答案”这句话，分别改写成：1）非常 polite 的学术表达；2）偏吐槽风格的表达；3）给 internal sharing 用的半口语表达。",
        "我有一个模糊目标：“把现有视频 RL 框架从单 GPU 拓展到多机多卡”。请你把它拆成 10 个可以在一周内逐个完成的具体小任务，每条控制在一行之内。",
        # 11–15 代码 / 工程
        "用自然语言解释下面这段伪代码可能在做什么，以及潜在的坑：for step in range(max_steps): sample_batch(); update_model(); if step % eval_interval == 0: evaluate_once()。尤其关注可能导致“线下 eval 很乐观、线上很崩”的原因。",
        "帮我设计一套“分布式训练日志”的基本规范，要求包含：日志分层（INFO/WARN/DEBUG）、每条日志至少有哪些字段、怎么避免多人同时看 log 时一头雾水。",
        "给一份“deepspeed / torchrun 多机训练经常卡住不动”时的排查 checklist，按优先级从高到低列 8 条，避免空泛建议，要具体到“看什么、怎么判定有问题”。",
        "设计一个简洁的“长视频推理服务 API”，分别给出：1）gRPC 接口定义（高层字段描述即可）；2）对应的 HTTP/JSON 请求体字段；要求能支持：上传视频 URL、可选文本问题、多轮对话 id。",
        "你现在是一个 QA，同事写了一个“视频事件抽取”模块。请给出 5 条测试用例描述，分别覆盖：正常情况、边缘长视频、无关背景噪音、缺帧、异常时间戳。",
        # 16–20 生活 / 决策
        "我在东京转机一晚：晚上 8 点到羽田，第二天下午从成田走，主要想购物。请列出 2–3 个完全不同风格的方案（比如：住银座 / 住新宿 / 住上野），每个方案列优点和潜在坑。",
        "我想建立一个“每天 30 分钟无干扰深度工作”的习惯，但目前工作节奏很碎。请给出一个为期 14 天的渐进式计划，每天一条任务，尽量现实而不是理想化。",
        "最近频繁出现“明知道要做，但就是不想动”的状态。请帮我从“动机、能量、环境、奖励”四个维度拆解可能原因，并给出每个维度 2 条可执行的小改动建议。",
        "把下面这句话改写成更委婉但边界清晰的表达：“这个需求现在插进来会严重影响我本周的计划，我不打算接。”要给出 3 个版本：对同事、对上级、对合作方（外部）。",
        "我工作日经常久坐、高压、睡眠不稳定，但又不太可能去健身房。请给出一个办公桌附近就能做的小动作列表（10 条以内），每条控制在一句话，主要目标是缓解肩颈紧张和脑子发胀。",
        # 21–25 推理 / 数学
        "有两个模型：A 训练数据少但干净，B 数据量大且噪声多。在参数量相同的前提下，想在小样本领域泛化更好，你会优先选哪种策略？请从 bias-variance 的视角解释。",
        "计算：从 1 加到 100 的和是多少？请同时给出两种不同的推导方法：一种直接算，一种利用公式，并解释公式为什么成立。",
        "我有一个策略，历史上胜率只有 40%，但盈亏比是 1:3（止损 1 单位，止盈 3 单位）。请帮我用期望的角度算一下长期来看是否正期望，并用非常直白的话解释给完全不懂数学的人听。",
        "举一个你认为在现实生活中“人类直觉经常错”的例子，要求不是那种网红谬误，而是日常工作相关的（如项目进度估计、多人协作沟通）。",
        "请构造一个需要至少 4 步推理才能得到结论的小故事逻辑题，并给出详细解答。要求故事背景和金融/交易无关，换成生活化场景。",
        # 26–30 创意 / 文本生成
        "写一段 120–150 字的文案，要求内容是“向团队道歉这周没完成所有目标”，但语气要混合：30% 真诚负责、30% 自嘲、40% 具体说明下周怎么补救。",
        "用两段短文，分别从“极度保守风控经理”和“激进 alpha trader”的角度，评论同一件事：某个新策略最近回测很好但实盘样本很少。每段不超过 120 字。",
        "把这句话翻译成英文，并做成像母语者写给同事的 Slack 消息：“今天 cluster 网络有点不稳定，训练可能会比预期慢一点，我会在结果出来后第一时间更新。”",
        "帮我设计一个适合测试“长视频推理能力”的场景脚本（比如一个 20 分钟的真实生活 vlog），要求包括：场景描述、关键事件节点、可能的问题类型（计数/顺序/因果）。",
        "你现在是一款还在开发中的长视频推理模型，请用第一人称写一段“自述”，内容包括：你最擅长的任务、你最容易犯错的地方、你希望人类怎么设计 prompt 让你表现更好。",
    ] * 500

    # 并发任务列表
    tasks = [
        infer_one(handle, p, idx)
        for idx, p in enumerate(prompts)
    ]

    # 并发跑 30 条
    results = await asyncio.gather(*tasks, return_exceptions=False)

    loop = asyncio.get_event_loop()

    loop.run_in_executor(None, lambda: Snapshot("deploy_rm/main/results").snapshot(results))

    for result in results:
        print(result)
        print("-" * 100)
    
    print(f"\n总共拿到 {len(results)} 条结果")


if __name__ == "__main__":
    ray.init(address="auto")
    st = time.time()
    asyncio.run(main())
    ed = time.time()
    print(f"Time taken: {ed - st} seconds")