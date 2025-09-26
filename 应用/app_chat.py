import argparse
import re
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PULSE_LABELS = ["沉脉", "迟脉", "虚脉", "弦脉", "浮脉", "细脉"]

SYSTEM_PROMPT = (
    "你是一个已经注入脉象知识的中文助手。"
    "当用户用 /pulse 指令提供两路脉搏波特征时，请依据提示只在"
    f"{PULSE_LABELS} 中选择最可能的一类，并给出简短理由。"
    "若没有 /pulse 指令，则进行普通多轮对话，保持简洁、技术准确。"
)

def build_pulse_prompt(ch1: str, ch2: str) -> str:
    return (
        f"已给出两路脉搏波特征，请在{PULSE_LABELS}中选择最可能的一类。"
        "优先输出一个标签（如“沉脉”），其后可用一两句解释原因。\n"
        f"通道一: {ch1}\n"
        f"通道二: {ch2}\n"
        "答案："
    )

def try_parse_pulse_command(text: str):
    """
    支持格式：
    /pulse 通道一=1,2,3 ; 通道二=4,5,6
    /pulse ch1=... ; ch2=...
    /pulse ch1:...  ch2:...
    返回 (is_pulse, prompt_str)
    """
    if not text.strip().startswith("/pulse"):
        return False, None

    # 尝试提取 ch1/ch2
    # 兼容 “通道一/二、ch1/ch2、=/: 分隔、; 分段”
    t = text.strip()[6:].strip()
    ch1 = ""
    ch2 = ""
    # 分片
    parts = re.split(r"[;；\n]+", t)
    for p in parts:
        if re.search(r"(通道一|ch1)\s*[:=]", p):
            ch1 = re.split(r"(通道一|ch1)\s*[:=]", p, maxsplit=1)[-1].strip()
        elif re.search(r"(通道二|ch2)\s*[:=]", p):
            ch2 = re.split(r"(通道二|ch2)\s*[:=]", p, maxsplit=1)[-1].strip()

    if not ch1 or not ch2:
        # 宽松再试：用逗号分两段
        m = re.match(r"[,，]?\s*(.+)\s*[,，;]\s*(.+)", t)
        if m:
            ch1, ch2 = m.group(1).strip(), m.group(2).strip()

    if ch1 and ch2:
        return True, build_pulse_prompt(ch1, ch2)
    return True, "指令解析失败：请使用格式 /pulse 通道一=<...>; 通道二=<...>"

def load_model_and_tokenizer(model_path: str, adapter_path: str = None):
    print("正在加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("正在加载基础模型 ...")
    # 使用 device_map='auto' 让 Accelerate 自动放置；4080 支持 bfloat16
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        device_map="auto",
    )

    if adapter_path and adapter_path != model_path:
        print("正在加载 LoRA 适配器 ...")
        model = PeftModel.from_pretrained(
            base,
            adapter_path,
            is_trainable=False
        )
    else:
        model = base

    model.eval()
    return tokenizer, model

def tokenize_chat(tokenizer, messages):
    """
    优先使用 chat_template；否则手动拼接。
    messages: [{"role":"system/user/assistant", "content": "..."}]
    """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # 兜底：简单拼接
        text = ""
        for m in messages:
            if m["role"] == "system":
                text += f"[系统]\n{m['content']}\n"
            elif m["role"] == "user":
                text += f"[用户]\n{m['content']}\n"
            else:
                text += f"[助手]\n{m['content']}\n"
        text += "[助手]\n"
    return text

class ChatEngine:
    def __init__(self, model_path, adapter_path=None, temperature=0.3, top_p=0.9, max_new_tokens=512):
        self.tokenizer, self.model = load_model_and_tokenizer(model_path, adapter_path)
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def reply(self, message, history):
        """
        history 是 [(user, assistant), ...]
        返回新的 assistant 文本
        """
        # 先判断是否是 /pulse 指令
        is_pulse, pulse_prompt = try_parse_pulse_command(message)

        if is_pulse and pulse_prompt:
            user_content = pulse_prompt
        elif is_pulse and pulse_prompt is None:
            user_content = "指令解析失败：请检查 /pulse 的通道参数格式。"
        else:
            user_content = message

        # 构造 chat messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for u, a in history:
            if u:
                messages.append({"role": "user", "content": u})
            if a:
                messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_content})

        prompt_text = tokenize_chat(self.tokenizer, messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated = output[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text

def build_ui(engine: ChatEngine):
    def respond(message, chat_history):
        answer = engine.reply(message, chat_history)
        chat_history = chat_history + [(message, answer)]
        return "", chat_history

    with gr.Blocks(fill_height=True, theme=gr.themes.Soft()) as demo:
        gr.Markdown("## Qwen 脉象对话助手（LoRA 推理）")
        gr.Markdown(
            "**用法**：\n"
            "• 普通聊天直接输入内容并发送。\n"
            "• 脉象判别：在输入框键入：\n"
            "```\n"
            "/pulse 通道一=147.9,1.38,-32.4,... ; 通道二=56.4,17.7,-7.0,...\n"
            "```\n"
        )

        chatbot = gr.Chatbot(height=520, placeholder="和我聊天，或用 /pulse 进行脉象判别")
        msg = gr.Textbox(placeholder="输入消息或 /pulse 指令", lines=4, autofocus=True)
        clear = gr.Button("清空对话")

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    return demo

def main():
    model_path = r"D:\qwen_pulse\models"       # 本地基座模型路径
    adapter_path = r"D:\qwen_pulse\qwen_pulse_lora"  # LoRA 权重路径（可选）
    port = 7860
    share = False   # True = 使用 gradio share

    engine = ChatEngine(model_path, adapter_path)
    ui = build_ui(engine)
    ui.launch(server_name="0.0.0.0", server_port=port, share=share, show_error=True)

if __name__ == "__main__":
    main()
