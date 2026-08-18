from typing import TypedDict

class ClassifyState(TypedDict, total=False):
    description: str          # 原始商品描述
    blocked: bool             # 输入护栏结果
    block_message: str
    info_sufficient: bool     # 信息完备性判断
    clarify_questions: list   # 需要追问的问题
    retrieved: list           # 检索结果
    llm_result: dict          # LLM 归类结论
    final: dict               # 最终输出
    trace: list               # 推理轨迹（面试演示用）