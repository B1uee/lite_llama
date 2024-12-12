import torch

from typing import List, Optional, Tuple, TypedDict
from transformers import AutoTokenizer

from .executor.model_executor import ModelExecutor
from .utils.file_interface import get_model_name_from_path
from .kernels.softmax_split import softmax_split

class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required


@torch.inference_mode()
def sample_top_p(probs, p):
    """
    执行 Top-p (Nucleus) 采样, 从概率分布中采样下一个词。
    
    参数：
        probs (torch.Tensor): 概率分布张量，形状为 `[batch_size, vocab_size]`。
        p (float): 累积概率阈值，取值范围在 0 到 1 之间。
    返回：
        torch.Tensor: 采样得到的词索引，形状为 `[batch_size, 1]`。

    说明：
        Top-p 采样算法: 选择概率累积和超过阈值 p 的最小集合，将这些词的概率重新归一化后进行采样。
    """
    # 对概率分布进行降序排序。probs_sort: 排序后的概率值，形状与 probs 相同。probs_idx: 排序后的索引，用于映射回原始词汇表。
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # 计算排序后概率的累积和. 返回的 probs_sum 是累积概率分布。
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # 保留累积概率未超过阈值 p 的词汇的概率，其余词汇的概率被置为 0.0。
    mask = probs_sum - probs_sort > p # 创建掩码，对于每个位置，计算累积概率（不包括当前词）是否超过阈值 p。
    probs_sort[mask] = 0.0 # 将累积概率超过阈值 p 的词的概率置零。

    # 对剩余的概率重新归一化, 确保总和为 1。
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # 从重新归一化的概率分布中采样下一个词. 返回的 next_token 是采样得到的词在排序后概率分布中的索引。
    next_token_sorted_idx = torch.multinomial(probs_sort, num_samples=1)
    # 在 probs_idx 的最后一维（dim=-1）中，使用 next_token_sorted_idx 作为索引，提取对应的值。沿着 dim=1（列）进行索引提取
    # NOTE: torch.gather 函数按照给定的索引张量 index，从输入张量中收集 (获取) 数据，并返回一个与索引张量形状一致的张量。
    next_token = torch.gather(probs_idx, -1, index = next_token_sorted_idx)
    
    return next_token # 返回采样得到的下一个词的索引

class GenerateText:
    """
    GenerateText 类用于加载LLaMA模型并执行迭代式生成式推理 (文本生成)。
    """

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
        assert self.model_config.vocab_size != -1, "Vocab size must be set"
        self.tokenizer = self.load_tokenizer(tokenizer_path)
    
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
        echo: bool = False,
        device = "cuda"
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        基于提供的提示词 (prompts) 使用语言生成模型生成文本序列。
        
        参数：
            prompt_tokens (List[List[int]]): 提示词的 token 序列，每个提示词是一个整数列表, 即 input_ids。
            max_gen_len (int): 最大生成序列长度。
            temperature (float, 可选): 控制采样随机性的温度值，默认 0.6。
            top_p (float, 可选): nucleus 采样的概率阈值，默认 0.9。
            echo (bool, 可选): 是否在输出中包含提示词，默认 False。
        返回：
            Tuple[List[List[int]], Optional[List[List[float]]]]: 生成的 token 序列和（可选）对应的 log 概率。
        """
        bsz = len(prompt_tokens) # 批量大小
        max_prompt_len = max(len(t) for t in prompt_tokens)
        total_len = min(self.model_config.max_seq_len, max_gen_len + max_prompt_len)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.model_executor.atten_info.max_actual_seq_len = max_prompt_len

        # 预分配tokens张量
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device = device)
        total_number_tokens = bsz * total_len

        # 填充提示词到 tokens 张量
        for seq_id, token_ids in enumerate(prompt_tokens):
            tokens[seq_id, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long, device = device)
        
        # 生成一个布尔张量，它的值为 True 的位置表示输入序列的实际内容（即非填充部分）, 形状为 (batch_size, total_len)
        input_text_mask = tokens != pad_id
        eos_reached = torch.zeros(bsz, dtype=torch.bool, device=device)
        prev_pos = 0 # 初始化上一次生成的位置

        # 一次性分配 bsz * total_len 个索引
        self.model_executor.atten_info.select_index = self.model_executor.kv_mem_manager.alloc_kvcache_index(total_number_tokens)
        select_index = self.model_executor.atten_info.select_index

        # 初始化每个批次项的序列长度
        actual_prompt_lens = torch.tensor([len(t) for t in prompt_tokens], dtype=torch.long, device=device)
        self.model_executor.atten_info.b_seq_len = actual_prompt_lens
        # print("self.model_executor.atten_info.b_seq_len ", self.model_executor.atten_info.b_seq_len)  

        # 初始化起始索引张量
        self.model_executor.atten_info.start_index = select_index[::total_len].to(torch.int32)
        # print("start_index: ", self.model_executor.atten_info.start_index)
        
        # 初始化当前已选择的批次项索引
        self.model_executor.atten_info.cur_select_index = select_index.unfold(0, max_prompt_len, total_len).reshape(-1)
        # print("Prefill stage cur_select_index: ", self.model_executor.atten_info.cur_select_index)
        
        for cur_pos in range(max_prompt_len, total_len):
            input_ids = tokens[:, prev_pos: cur_pos] # 当前输入 token ids, decode 阶段 input_ids shape is [4, 1]         
            logits = self.model_executor.forward(input_ids, prev_pos) # 模型执行器的前向推理, logits shape is [batch_size, shape, vocab_size]
            
            if prev_pos > 0:
                self.model_executor.atten_info.max_actual_seq_len += 1
                self.model_executor.atten_info.b_seq_len += 1
            
            self.model_executor.atten_info.cur_select_index = (self.model_executor.atten_info.start_index 
                                                               + self.model_executor.atten_info.b_seq_len)
            
            # print("Decode stage cur_select_index: ", self.model_executor.atten_info.cur_select_index)

            probs = softmax_split(logits[:, -1] / temperature) # torch.softma 将 logits 转换为概率分布。
            next_token = sample_top_p(probs, top_p) # next_token 形状为 [batch_size, 1]
            next_token = next_token.reshape(-1) # 调整为一维, shape is batch_size

            # 仅替换生成部分的token
            mask = ~input_text_mask[:, cur_pos]
            next_token = torch.where(mask, next_token, tokens[:, cur_pos])
            tokens[:, cur_pos] = next_token
            
            # 更新结束标志
            eos_reached |= (mask & (next_token == self.tokenizer.eos_token_id))
            prev_pos = cur_pos

            if eos_reached.all(): # 如果所有样本均到达结束 token，停止生成
                break
        
        # out_tokens = self.process_output_tokens(tokens, prompt_tokens, max_gen_len, echo, self.tokenizer.eos_token_id)
        self.model_executor.kv_mem_manager.release_ref(select_index) # 减少 kv cache 内存管理器的引用计数

        return tokens
    
    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        echo: bool = False,
        device = "cuda",
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.
        """
        input_ids = self.tokenizer.batch_encode_plus(prompts, add_special_tokens=True).input_ids
        generated_ids = self.generate(
            prompt_tokens = input_ids,
            max_gen_len = max_gen_len,
            temperature = temperature,
            top_p = top_p,
            echo = echo,
            device = device,
        )

        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts
    
    def process_output_tokens(
        self,
        tokens: torch.Tensor,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        echo: bool,
        eos_token_id,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        处理生成的 tokens 和对应的对数概率，提取最终的输出序列。
        """
        out_tokens = []

        for i, seq_tokens in enumerate(tokens.tolist()): # 将 tokens 转换为列表
            prompt_len = len(prompt_tokens[i])
            # 根据是否需要在输出中包含提示词，确定起始位置
            start_idx = 0 if echo else prompt_len
            end_idx = prompt_len + max_gen_len
            # 截取从起始位置到最大生成长度的 tokens
            generated_toks = seq_tokens[start_idx:end_idx]
            # 检查是否存在结束符，若存在则截断到结束符之前
            if eos_token_id in generated_toks:
                eos_idx = generated_toks.index(eos_token_id)
                generated_toks = generated_toks[:eos_idx]

            out_tokens.append(generated_toks)

        return out_tokens
