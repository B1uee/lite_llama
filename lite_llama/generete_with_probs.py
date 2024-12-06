from typing import Optional
import torch
import logging
from typing import List, Literal, Optional, Tuple, TypedDict
import torch.nn.functional as F 
from transformers import AutoTokenizer

from .executor.model_executor import ModelExecutor
from .utils.file_interface import get_model_name_from_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]
    logprobs: List[float]

class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]
    logprobs: List[float]

Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

@torch.inference_mode()
def sample_top_p(probs, p: float):
    # 使用 in-place 操作减少内存分配
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # mask 记录累积概率超过 p 的位置
    mask = probs_sum - probs_sort > p
    # 将超过 p 的概率置 0
    probs_sort[mask] = 0.0
    # 原地归一化
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token_sorted_idx = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token_sorted_idx)
    return next_token

class GenerateText:
    def __init__(self, 
        checkpoints_dir: str,
        tokenizer_path: str,
        max_seq_len = 1024,
        max_gpu_num_blocks = None,
        load_model = True,
        triton_weight = True,
        compiled_model = False,
        device="cuda",
    ):
        self.checkpoints_dir = checkpoints_dir
        self.compiled_model = compiled_model

        self.model_executor = ModelExecutor.build(
            checkpoints_dir = checkpoints_dir,
            load_model = load_model,
            max_gpu_num_blocks = max_gpu_num_blocks,
            max_seq_len = max_seq_len,
            triton_weight = triton_weight,
            device = device
        )
        self.model_config = self.model_executor.model_config
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.device = device
    
    def load_tokenizer(self, pretrained_model_name_or_path):
        model_name = get_model_name_from_path(pretrained_model_name_or_path)
        # 根据模型名称决定是否使用 fast tokenizer
        use_fast = True
        if 'llava' in model_name.lower():
            use_fast = False
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=use_fast)
        return tokenizer
    
    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = True,
        echo: bool = False,
        device = "cuda"
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        model_config = self.model_config
        bsz = len(prompt_tokens)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= model_config.max_seq_len
        total_len = min(model_config.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        for seq_id, token_ids in enumerate(prompt_tokens):
            length = len(token_ids)
            tokens[seq_id, :length] = torch.tensor(token_ids, dtype=torch.long, device=device)

        input_text_mask = tokens != pad_id
        eos_reached = torch.zeros(bsz, dtype=torch.bool, device=device)

        token_logprobs = torch.zeros((bsz, total_len), dtype=torch.float, device=device) if logprobs else None

        if min_prompt_len == total_len and logprobs:
            # 若无生成空间，直接计算已存在 tokens 的对数似然（可选）
            logits = self.model_executor.forward(tokens, 0)
            token_logprobs.copy_(
                -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens,
                    reduction="none",
                    ignore_index=pad_id,
                )
            )

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # 仅在需要生成时进入循环
        token_count = 0
        prev_pos = 0
        # 为减少循环中的判断逻辑，这里将range起点由min_prompt_len改成max_prompt_len
        # 因为在[min_prompt_len, max_prompt_len)区间，有的样本已经完成输入填充，有的刚好开始生成
        # 实际上可以统一从 max_prompt_len 开始生成，因为在 (min_prompt_len, max_prompt_len) 的位置上，有的样本还属于prompt部分
        # 这样减少复杂判断逻辑
        gen_start = max_prompt_len
        for cur_pos in range(gen_start, total_len):
            # 取出新加入模型的 token (一般为单步输入)
            input_ids = tokens[:, cur_pos - 1: cur_pos] if cur_pos > 0 else tokens[:, :1]
            logits, select_index = self.model_executor.forward(input_ids, prev_pos)
            prev_pos = cur_pos

            # 对最后一个位置进行 softmax
            step_logits = logits[:, -1, :]
            if temperature > 0:
                probs = F.softmax(step_logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p).reshape(-1)
            else:
                next_token = torch.argmax(step_logits, dim=-1)

            # 对仍在生成过程（非输入部分）的位置写入next_token
            # 对尚在prompt部分的位置保持原值不变
            to_generate = ~input_text_mask[:, cur_pos]
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            if logprobs:
                # 本步仅计算新生成 token 的 logprob，而非整段重复计算
                # 使用CrossEntropy时，只需对单一token位置进行计算, 可用gather简化
                # CE = -log(softmax)，logit对应step_logits
                # 获取实际的token目标值
                target = tokens[:, cur_pos]
                # 使用 log_softmax 代替 cross_entropy，可以只提取相应token的logprob
                log_probs = F.log_softmax(step_logits, dim=-1)
                step_logprobs = torch.gather(log_probs, 1, target.unsqueeze(1)).squeeze(1)

                # 将计算结果写入相应位置
                token_logprobs[:, cur_pos] = step_logprobs

            # 检查终止条件
            finished = to_generate & (next_token == self.tokenizer.eos_token_id)
            eos_reached |= finished

            token_count += (to_generate).sum().item()
            if eos_reached.all():
                break

        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_sec = start_event.elapsed_time(end_event) / 1000.0
        tokens_per_second = token_count / elapsed_time_sec if elapsed_time_sec > 0 else float('inf')
        logger.info(f"Batch inference time, no decode: {elapsed_time_sec * 1000:.4f} ms")
        logger.info(f"Tokens per second, no decode: {tokens_per_second:.2f} tokens/s")

        out_tokens, out_logprobs = self.process_output_tokens(
            tokens, prompt_tokens, max_gen_len, logprobs, echo, self.tokenizer.eos_token_id, token_logprobs
        )

        # 减少 kv cache 内存管理器的引用计数
        # 若最后一次生成失败，select_index可能为上一次保留值，这里加个判断
        if 'select_index' in locals():
            self.model_executor.kv_mem_manager.release_ref(select_index)

        return out_tokens, out_logprobs

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
        device = "cuda",
    ) -> List[CompletionPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model_config.max_seq_len - 1

        input_ids = self.tokenizer.batch_encode_plus(prompts, add_special_tokens=True).input_ids
        generated_ids, generation_logprobs = self.generate(
            prompt_tokens = input_ids,
            max_gen_len = max_gen_len,
            temperature = temperature,
            top_p = top_p,
            logprobs = logprobs,
            echo = echo,
            device = device,
        )

        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t, skip_special_tokens=True),
                    "tokens": [self.tokenizer.decode([x], skip_special_tokens=True) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generated_ids, generation_logprobs)
            ]
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts

    def process_output_tokens(
        self,
        tokens: torch.Tensor,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        logprobs: bool,
        echo: bool,
        eos_token_id,
        token_logprobs: Optional[torch.Tensor] = None
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:

        out_tokens = []
        out_logprobs = [] if logprobs else None
        tokens_list = tokens.tolist()  # 转为CPU列表，只在最终处理输出时进行

        if logprobs:
            logprobs_list = token_logprobs.tolist()

        for i, seq_tokens in enumerate(tokens_list):
            prompt_len = len(prompt_tokens[i])
            start_idx = 0 if echo else prompt_len
            end_idx = prompt_len + max_gen_len
            generated_toks = seq_tokens[start_idx:end_idx]

            if logprobs:
                seq_logprobs = logprobs_list[i][start_idx:end_idx]

            # 截断到 EOS 之前
            if eos_token_id in generated_toks:
                eos_idx = generated_toks.index(eos_token_id)
                generated_toks = generated_toks[:eos_idx]
                if logprobs:
                    seq_logprobs = seq_logprobs[:eos_idx]

            out_tokens.append(generated_toks)
            if logprobs:
                out_logprobs.append(seq_logprobs)

        return (out_tokens, out_logprobs if logprobs else None)
    
    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model_config.max_seq_len - 1
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} "
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
            )
            prompt_tokens.append(dialog_tokens)

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t, skip_special_tokens=True)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode([x], skip_special_tokens=True) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t, skip_special_tokens=True) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]