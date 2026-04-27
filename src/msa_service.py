import queue
import pickle
import json
from tqdm import tqdm
import torch
import torch.distributed as dist
import multiprocessing as mp
import threading
import os
import numpy as np
from typing import Tuple, List, Optional, Dict, Union, Any, Callable
from dataclasses import dataclass
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoConfig

from src.config.memory_config import GenerateConfig, MemoryConfig, ModelConfig
from src.prefill import PrefillStage1Worker
from src.types import ProtocolConstants
from src.utils.cache import create_cache, CustomDynamicCache
from src.utils.gpu_worker import GpuWorker
from src.utils.tools import RequestLimiter, format_bytes, cumulative_concat, compose_input
from src.msa.model import MSAForCausalLM, MSAConfig

# ==========================================
# 数据结构定义
# ==========================================

MEMORY_WORKER_READY = "MEMORY_WORKER_READY"
MEMORY_WORKER_CLOSE = "MEMORY_WORKER_CLOSE"
MEMORY_WORKER_BLOCKS = "MEMORY_WORKER_BLOCKS"
MEMORY_WORKER_IDX_TO_DOC = "MEMORY_WORKER_IDX_TO_DOC"
MEMORY_WORKER_REPORT_KV_STATES = "MEMORY_WORKER_REPORT_KV_STATES"


def _resolve_v_dtype(k_dtype: torch.dtype) -> torch.dtype:
    """V-cache dtype, controlled by MSA_V_DTYPE env var. Default mirrors K
    dtype (BF16 typically). 'fp8' switches V to float8_e4m3fn for ~2x cache
    halving (DSV4 borrow). K stays at k_dtype since routing reads K directly."""
    v = os.environ.get("MSA_V_DTYPE", "").lower()
    if v in ("fp8", "float8", "float8_e4m3fn"):
        return torch.float8_e4m3fn
    return k_dtype

@dataclass
class CmdBase:
    name: str = ""

@dataclass
class ResultBase:
    name: str = ""
    gpu_id: int = 0


@dataclass
class QueryTemplateKVPrefixCmd(CmdBase):
    layer_idx: int = -1 # -1 indicate get all layers
    
    def __post_init__(self):
        self.name = "query_template_kv_prefix"

@dataclass
class QueryTemplateKVPrefixResult(ResultBase):
    template_kvcache: Dict[int, Dict[str, torch.Tensor]] = None
    
    def __post_init__(self):
        self.name = "query_template_kv_prefix"

@dataclass
class PrefillStage2Cmd(CmdBase):
    query_states: torch.Tensor = None
    attention_mask: torch.Tensor = None
    doc_ids: torch.Tensor = None
    layer_index: int = 0
    
    def __post_init__(self):
        self.name = "prefill_stage2"

@dataclass
class PrefillStage2Result(ResultBase):
    key: Union[torch.Tensor, List[torch.Tensor]] = None
    value: Union[torch.Tensor, List[torch.Tensor]] = None
    pooled_doc_ids: Union[torch.Tensor, List[torch.Tensor]] = None
    
    def __post_init__(self):
        self.name = "prefill_stage2"

# TODO: 这个可以取消，因为这是为了让prefill产生的 pooled doc id 全局唯一，但是这个完全可以
#       在归集stage2结果的时候才去做，只要用 block 或者 gpu id 做一个偏移就好
@dataclass
class SetDocIDCmd(CmdBase):
    pooled_doc_id_bias: int = 0
    
    def __post_init__(self):
        self.name = "set_doc_id"

@dataclass
class ReportDocID(ResultBase):
    pooled_doc_id_bias: int = 0

    def __post_init__(self):
        self.name = "report_doc_id"

@dataclass
class AddDocumentsCmd(CmdBase):
    docs: List = None  # List[Document] assigned to this worker
    idx_to_doc_update: Dict = None  # new {doc_id: text} entries

    def __post_init__(self):
        self.name = "add_documents"

@dataclass
class AddDocumentsResult(ResultBase):
    doc_ids: List[int] = None  # global doc_ids of newly added documents

    def __post_init__(self):
        self.name = "add_documents"

@dataclass
class ResetMemoryCmd(CmdBase):
    docs: List = None  # List[Document] assigned to this worker (this GPU's bucket)
    idx_to_doc: Dict = None  # full {doc_id: text} mapping

    def __post_init__(self):
        self.name = "reset_memory"

@dataclass
class ResetMemoryResult(ResultBase):
    ok: bool = False
    error: str = None

    def __post_init__(self):
        self.name = "reset_memory"

class CustomDynamicCacheOnCPU(CustomDynamicCache):
    def __init__(self, _distributed_cache_data=None):
        super().__init__(_distributed_cache_data)

    # def record_kwargs(self, layer_idx, kwargs):
    #     d = {}
    #     for k, v in kwargs.items():
    #         if v is not None and torch.is_tensor(v):
    #             d[k] = v.cpu() if v.is_cuda else v.clone()
    #         else:
    #             d[k] = v
    #     super().record_kwargs(layer_idx, kwargs)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        # if key_states is not None and torch.is_tensor(key_states) and key_states.is_cuda:
        #     key_states = key_states.cpu()
        if value_states is not None and torch.is_tensor(value_states) and value_states.is_cuda:
            value_states = value_states.cpu()
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

@dataclass
class GenerateRequest(CmdBase):
    # 每次用户发送一个 generate 请求，我们都分配一个 msgid
    # 但是我们会将其分成world个GenerateRequest，seq_id表示其自请求顺序
    msg_id: int = 0
    seq_id: int = 0
    input_ids: List[int] = None
    attention_mask: List[int] = None
    doc_ids: List[int] = None
    positions: List[int] = None
    require_recall_topk: bool = False
    prompts: List[str] = None
    
    def __post_init__(self):
        self.name = "generate_request"

@dataclass
class GenerateResponse():
    msg_id: int = -1
    seq_id: int = -1
    gpu_id: int = 0
    generated_texts: List[str] = None
    recall_topk: Dict[int, List] = None


# response receiver: callback(generated_texts, required_recall_topk, userdata)
GenerateReceiver = Callable[[List[str], Any, Any], None]

@dataclass
class GenerateStub:
    msg_id: int = 0
    userdata: Any = None
    responses: List[GenerateResponse] = None
    nr_dummy: int = 0 # 标记这个 batch 尾部有几个query 是填充的空请求
    callback: Optional[GenerateReceiver] = None

    def respond(self):
        if self.callback is None:
            print("no callback")
            return

        self.responses.sort(key=lambda r: r.seq_id)
        txts = [] 
        recall_topk: Dict[int, List] = {}
        for rsp in self.responses:
            txts.extend(rsp.generated_texts)
            if rsp.recall_topk:
                for layer, topks in rsp.recall_topk.items():
                    if layer not in recall_topk:
                        recall_topk[layer] = []
                    recall_topk[layer].extend(topks)
        
        self.callback(txts[:len(txts)-self.nr_dummy], recall_topk, self.userdata)
            
                
            
        

@dataclass
class Document:
    doc: str = ""
    doc_id: int = 0
    num_chunks: int = 0

@dataclass
class BlockModelInput:
    doc_input_ids: torch.Tensor
    doc_attention_mask: torch.Tensor
    doc_ids: torch.Tensor
    position_ids: torch.Tensor
    num_chunks: int
    chunk_sizes: List[int]

@dataclass
class BlockDesc:
    """一个 block 的描述数据"""
    docs: List[Document] = None # 所有分配到此 GPU 上的文档
    tmp_doc_ids: List[torch.Tensor] = None # 临时存储

    # kvcache的每一个 chunk 分组编号，从 1 开始，形如[1,1,1,2,2,3,3,3,...]
    # pool_ids[-1] 就是文档总数，相同的编号对应的 chunks 表示同属于同一个文档
    pool_ids: torch.Tensor = None
    # 用于通过 pool_id查找 global doc id，即 global_doc_ids = doc_ids[pool_ids]
    doc_ids: torch.Tensor = None
    doc_ids_cpu: torch.Tensor = None # doc_ids 的 CPU 版本，方便后续索引使用

    # 用 local id，也就是pool_ids中的 id 来进行检索
    doc_offsets_cpu: torch.Tensor = None # 每个文档 chunk 的起始索引 [nr_docs + 1]
    doc_lens_cpu: torch.Tensor = None    # 每个文档包含的 chunk 数量 [nr_docs + 1]

    def init_docs(self, docs: List[Document], device):
        self.docs = docs
        self.doc_ids_cpu = torch.LongTensor([0] + [item.doc_id for item in self.docs])
        self.doc_ids = self.doc_ids_cpu.to(device)
        self.tmp_doc_ids = []

        # 计算偏移量：假设 doc 0 长度为 0
        lens = [0] + [doc.num_chunks for doc in docs]
        self.doc_lens_cpu = torch.LongTensor(lens)
        self.doc_offsets_cpu = torch.cumsum(torch.LongTensor([0] + lens[:-1]), dim=0)

    def create_global_doc_ids(self, local_doc_ids):
        return self.doc_ids[local_doc_ids]

    def merge_poolig_doc_id(self, device):
        self.pool_ids_cpu = cumulative_concat(self.tmp_doc_ids)
        self.pool_ids = self.pool_ids_cpu.to(device)
        self.tmp_doc_ids.clear()

    def chunks(self):
        return sum(doc.num_chunks for doc in self.docs)
    
    @property
    def nr_docs(self):
        return len(self.docs)

@dataclass
class BlockData:
    """本 GPU 上的文档 kvcache：[1, num_head, num_chunks, head_dim]"""
    k: torch.Tensor = None # k 用于 attn 计算，如果 rk 不存在则保存于 GPU，同时用于检索，否则保存在 CPU
    v: torch.Tensor = None # 存储于 CPU
    rk: torch.Tensor = None # 用于检索，此时我们有独立的检索 k，需要保存于 GPU

    def get_router_k(self):
        """获取用于检索的k"""
        return self.rk if self.rk is not None else self.k
    

@dataclass
class SliceDesc:
    """block分片信息，每一个分片包含了一部分文档信息
    """
    nr_docs: int = 0 # 文档数量
    nr_chunks: int = 0 # chunk数量
    global_doc_ids: torch.Tensor = None # 文档 ID，shape 为[nr_chunks]

    # 下面这几个 tensor 是我们预先计算好，用于 prefill stage2 的数据
    local_doc_ids: torch.Tensor = None
    local_doc_ids_0: torch.Tensor = None
    original_doc_ids: torch.Tensor = None


class MemoryClientBase():
    """被模型调用的接口"""
    def get_template_prefix_kvcaches(self, layer_idx: int):
        pass
    def doc_query(self, query_states: torch.Tensor, attention_mask: torch.Tensor, layer_idx: int):
        pass

# ==========================================
# Service 实现
# ==========================================
class Memory(GpuWorker):
    """Memory工作进程"""

    def __init__(self,
                 gpu_id: int,
                 generate_config: GenerateConfig,
                 model_config: ModelConfig,
                 memory_config: MemoryConfig):
        """
        该 worker 被MemoryWorker创建并仅执行prefill stage 1获取block 的kv cache
        """
        super().__init__(gpu_id, model_config.get_model_envs())

        self.model_config = model_config
        self.generate_config = generate_config
        self.memory_config = memory_config

        self.msa_model_config = MSAConfig.from_pretrained(model_config.model_path)

        self.router_layer_ids = list(range(self.num_model_layers()) if model_config.router_layer_idx == "all"
                                     else map(int, model_config.router_layer_idx.split(",")))
        self.router_layer_ids.sort()

        self.num_key_value_groups = self.msa_model_config.num_attention_heads // self.msa_model_config.num_key_value_heads


        self.vcache_in_cpu = True
        self.v_device = 'cpu' if self.vcache_in_cpu else self.device

        ######## 下面几个定义是可以动态加载进来的
        # 每一个router层上的分片数据列表: layer_idx => [block 1, block 2, ...]
        self.blocks: Dict[int, BlockData] = {layer: BlockData() for layer in self.router_layer_ids}
        # 每一个 block 的描述数据, [block 1 desc, block 2 desc, ...]
        self.block_desc: BlockDesc = BlockDesc()
        # 每一层上的template prefix kv cache:  layer idx -> (k, v)
        self.template_prefix_kvcache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        # 对 block 进行chunk 分割时，每个 block 大小
        self.slice_chunk_size = memory_config.slice_chunk_size
        # 为了后续计算方便，我们预先处理好了 k slice，shape 为[1, num_kv_heads, 1, hdim, chunk]
        self.k_slices: Dict[int, List[torch.Tensor]] = []
        self.slice_desc: List[SliceDesc] = []

        self.head_reduce_method = self.msa_model_config.msa_config.head_reduce_method
        self.query_reduce_method = self.msa_model_config.msa_config.query_reduce_method
        self.chunk_reduce_method = self.msa_model_config.msa_config.chunk_reduce_method
        self.aux_loss_method = self.msa_model_config.msa_config.aux_loss_method
        self.decouple_router = self.msa_model_config.msa_config.decouple_router

        if self.aux_loss_method == "INFONCE" and self.decouple_router:
            self.scaling = -1.0
        else:
            self.scaling = self.msa_model_config.head_dim**-0.5

    def _start_worker(self):
        if getattr(self, "_worker_process", None) is not None:
            return
        self._worker_req_q, self._worker_rsp_q = mp.Queue(), mp.Queue()

        self._worker_process = mp.Process(
            target=PrefillStage1Worker.prefill_worker_main,
            args=(self.gpu_id, self._worker_req_q, self._worker_rsp_q,
                  self.model_config.model_path,  self.generate_config.template,
                  self.memory_config.pooling_kernel_size, self.memory_config.block_size,
                  self.model_config.get_model_envs()),
            name=f"PrefillStage1Worker-{self.gpu_id}",
            daemon=True,
        )
        self._worker_process.start()

        PrefillStage1Worker.wait_for_ready(self._worker_rsp_q)
        print(f"Prefill worker {self.gpu_id} is ready")

    def _stop_worker(self):
        if getattr(self, "_worker_process", None) is None:
            return
        print(f"Stop prefill worker on GPU {self.gpu_id}")
        PrefillStage1Worker.close_worker(self._worker_req_q)
        self._worker_process.join()
        self._worker_process = None
        self._worker_req_q, self._worker_rsp_q = None, None


    def num_model_layers(self):
        return self.msa_model_config.num_hidden_layers
    
    
    def serialize(self, path: str):
        """
        安全分片序列化：
        1. 创建目录
        2. 保存元数据 (meta.pt)
        3. 逐层保存 Tensor 到独立文件，避免内存峰值
        """
        # 1. 准备目录 (如果存在则警告或清空，这里选择清空以确保一致性)
        if os.path.exists(path):
            print(f"Warning: Directory {path} exists. Overwriting data inside.")
        else:
            os.makedirs(path, exist_ok=True)
        
        stub = {
            "memory_file_path": self.memory_config.memory_file_path,
            "world": self.generate_config.world,
        }
        with open(os.path.join(path, "stub.json"), "w") as fp:
            json.dump(stub, fp)

        # 2. 保存元数据 (BlockDesc + 结构信息)
        # 注意：BlockData 里的 heavy tensor 不在这里存，只存结构
        meta_dict = {
            "router_layer_ids": self.router_layer_ids,
            "block_desc": {
                "docs": self.block_desc.docs,
                "doc_ids": self.block_desc.doc_ids.cpu(),
                # pool_ids 转 CPU 保存
                "pool_ids": self.block_desc.pool_ids.cpu() if self.block_desc.pool_ids is not None else None
            },
        }
        torch.save(meta_dict, os.path.join(path, "meta.pt"))

        # 3. 逐层保存 Blocks (重头戏)
        print("Serializing Blocks (Sharded)...")
        for layer_id, block in tqdm(self.blocks.items(), desc="Saving Layers"):
            # 分离存储 K 和 V
            if block.k is not None:
                # K 是 GPU -> CPU
                torch.save(block.k.cpu(), os.path.join(path, f"layer_{layer_id}_k.pt"))
            
            if block.v is not None:
                # V 已经在 CPU，直接存
                # 提示：对于 CPU 上的极大 Tensor，torch.save 效率很高
                torch.save(block.v.cpu(), os.path.join(path, f"layer_{layer_id}_v.pt"))

            if block.rk is not None:
                torch.save(block.rk.cpu(), os.path.join(path, f"layer_{layer_id}_rk.pt"))

        # 4. 保存 Prefix Cache
        if self.template_prefix_kvcache:
            print("Serializing Prefix Cache...")
            prefix_data = {}
            for layer_id, (k, v) in self.template_prefix_kvcache.items():
                prefix_data[layer_id] = (k.cpu(), v.cpu())
            torch.save(prefix_data, os.path.join(path, "prefix_cache.pt"))

        print(f"Serialization complete. Data saved to {path}")

    def deserialize(self, path: str):
        """
        流式反序列化：
        每次只加载一个文件进内存，处理完后立即释放或转移到 GPU，
        将系统内存（RAM）占用控制在最低。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist.")
        
        print("Deserialize from ", path)

        # 1. 加载元数据
        meta_path = os.path.join(path, "meta.pt")
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        
        # 恢复 BlockDesc
        bd_info = meta["block_desc"]
        self.block_desc = BlockDesc(
            docs=bd_info["docs"],
            doc_ids=bd_info["doc_ids"].to(self.device),
            pool_ids=bd_info["pool_ids"].to(self.device) if bd_info["pool_ids"] is not None else None
        )

        # 2. 逐层恢复 Blocks
        # 关键：我们不一次性加载所有文件，而是循环加载
        self.blocks = {}
        print(f"Deserializing Blocks to GPU {self.gpu_id}...")
        
        for layer_id in tqdm(self.router_layer_ids, desc=f"GPU-{self.gpu_id} Loading Layers", position=self.gpu_id):
            k_path = os.path.join(path, f"layer_{layer_id}_k.pt")
            v_path = os.path.join(path, f"layer_{layer_id}_v.pt")
            rk_path = os.path.join(path, f"layer_{layer_id}_rk.pt")
            
            current_k = None
            current_v = None
            current_rk = None

            # 加载 K -> 立即转 GPU -> 删除 CPU 副本
            if os.path.exists(k_path):
                temp_k = torch.load(k_path, map_location="cpu", weights_only=False)
                current_k = temp_k.to(self.device) # 移入显存
                del temp_k # 立即释放内存

            # 加载 V -> 驻留 CPU
            if os.path.exists(v_path):
                # 既然 V 是 CPU 存储且极大，直接 load 即可
                # 如果 V 极其巨大（比如超过 RAM），可以考虑 mmap=True，
                # 但这要求保存时格式兼容，且 V 通常作为 cache 需要频繁读取，RAM 还是最快的
                current_v = torch.load(v_path, map_location="cpu", weights_only=False)

            if os.path.exists(rk_path):
                current_rk = torch.load(rk_path, map_location="cpu", weights_only=False)

            self.blocks[layer_id] = BlockData(k=current_k, v=current_v, rk=current_rk)
            
            # 显式触发垃圾回收（可选，但在内存吃紧时有用）
            # import gc; gc.collect() 

        # 3. 恢复 Prefix Cache
        prefix_path = os.path.join(path, "prefix_cache.pt")
        if os.path.exists(prefix_path):
            prefix_raw = torch.load(prefix_path, map_location="cpu", weights_only=False)
            self.template_prefix_kvcache = {}
            for layer_id, (k_cpu, v_cpu) in prefix_raw.items():
                self.template_prefix_kvcache[layer_id] = (
                    k_cpu.to(self.device),
                    v_cpu.to(self.device)
                )
        
        print("Deserialization complete.")
    
    def _init_block_data(self, shape: torch.Size, dtype: torch.dtype, has_rk=False):
        total_chunks = self.block_desc.chunks()
        v_dtype = _resolve_v_dtype(dtype)
        for layer_idx in self.router_layer_ids:
            block_data = self.blocks[layer_idx]

            if block_data.k is None:
                kv_shape = list(shape)
                kv_shape[2] = total_chunks
                # TODO: deserialize 时需要考虑 pin_memory
                block_data.k = torch.empty(kv_shape, device=self.v_device if has_rk else self.device, dtype=dtype, pin_memory=has_rk)
                # SSD mmap offload: V cache backed by disk file, OS manages hot/cold pages
                _mmap_dir = os.environ.get("MSA_KV_CACHE_DIR", "")
                if _mmap_dir:
                    os.makedirs(_mmap_dir, exist_ok=True)
                    # numpy lacks bf16/fp8; pick byte-equivalent storage and view back as torch dtype
                    if v_dtype == torch.float8_e4m3fn:
                        _store_dtype = np.uint8
                    elif v_dtype == torch.bfloat16:
                        _store_dtype = np.uint16
                    elif v_dtype == torch.float16:
                        _store_dtype = np.float16
                    else:
                        _store_dtype = np.float32
                    _mmap_path = os.path.join(_mmap_dir, f"block_{id(block_data)}_layer_{layer_idx}_v.bin")
                    _mmap = np.memmap(_mmap_path, dtype=_store_dtype, mode='w+', shape=tuple(kv_shape))
                    block_data.v = torch.from_numpy(_mmap).view(v_dtype)
                    block_data._v_mmap_path = _mmap_path
                else:
                    block_data.v = torch.empty(kv_shape, device=self.v_device, dtype=v_dtype, pin_memory=True)
                block_data.rk = torch.empty(kv_shape, device=self.device, dtype=dtype) if has_rk else None

    def save_idx_to_doc(self, idx_to_doc: Dict[int, str]):
        self.idx_to_doc = idx_to_doc
        
    def generate_blocks(self, docs: List[Document]):
        memory_data_path = os.environ.get("MEMORY_DATA_PATH", "")
        if memory_data_path:
            memory_data_path = os.path.join(memory_data_path, f"gpu{self.generate_config.world}", str(self.gpu_id))
            if os.path.exists(memory_data_path):
                self.deserialize(memory_data_path)
                assert len(docs) == self.block_desc.nr_docs
                self._post_process()
                return

        self._start_worker()

        self.block_desc.init_docs(docs, self.device)

        PrefillStage1Worker.send_documents(self._worker_req_q, docs)

        kv_offset = 0
        recv = 0

        total_docs = len(docs)
        total_chunks = self.block_desc.chunks()
        pbar = tqdm(total=total_docs, desc=f"Worker-{self.gpu_id} memory inference", position=self.gpu_id)
        while recv < total_docs:
            meta = PrefillStage1Worker.recv_meta(self._worker_rsp_q)

            recv += meta['nr_docs'] # 此 meta 包含了几个文档
            pbar.update(meta['nr_docs'])

            past_key_values: CustomDynamicCacheOnCPU = meta["past_key_values"]

            # 仅保存一次
            if not self.template_prefix_kvcache:
                kwargs = past_key_values.cache_kwargs
                self.template_prefix_kvcache = {
                    layer_idx: (
                        kwargs[layer_idx]['template_prefix_kcache'].to(self.device),
                        kwargs[layer_idx]['template_prefix_vcache'].to(self.device)
                        )
                    for layer_idx in range(self.num_model_layers())
                }


            meta_chunks = 0 # 这次的 meta 有多少 chunks

            for layer_idx in self.router_layer_ids:
                block_data = self.blocks[layer_idx]

                k, v = past_key_values.get_kvcache(layer_idx) # for attn calc
                rk = past_key_values.get_router_kcache(layer_idx) # for router, if any

                desc: BlockDesc = self.block_desc
                if block_data.k is None:
                    self._init_block_data(k.shape, k.dtype, has_rk=rk is not None)

                if layer_idx == self.router_layer_ids[0]:
                    # 聚合 pooling doc id，因为每一层上的 pooling doc id 都是一样的，所以只聚合第一层
                    desc.tmp_doc_ids.append(past_key_values.cache_kwargs[layer_idx]['pooled_doc_ids'])
                    
                k_cache, v_cache, rk_cache = block_data.k, block_data.v, block_data.rk
                if meta_chunks == 0:
                    meta_chunks = k.shape[2]
                else:
                    assert meta_chunks == k.shape[2], f"expect meta_chunks {meta_chunks} but got {k.shape[2]}"
                    assert kv_offset + meta_chunks <= total_chunks, f"overflow: offset {kv_offset}, meta_chunks {meta_chunks}, total_chunks {total_chunks}"

                # print(f"shape {k.shape} offset {kv_offset}")
                k_cache[:, :, kv_offset:kv_offset+meta_chunks] = k.to(k_cache.device)
                v_cache[:, :, kv_offset:kv_offset+meta_chunks] = v.to(v_cache.device).to(v_cache.dtype)
                if rk is not None:
                    rk_cache[:, :, kv_offset:kv_offset+meta_chunks] = rk.to(rk_cache.device)

            # 每次一个 meta 处理完才需要进偏移
            kv_offset += meta_chunks

            past_key_values.clear_kvcache()

        pbar.close()
        # persistent worker: signal end-of-batch, leave alive for reuse (daemon=True ensures auto-kill on parent exit)
        PrefillStage1Worker.recv_batch_done(self._worker_rsp_q)

        # 合并临时pooled doc id
        self.block_desc.merge_poolig_doc_id(self.device)

        if memory_data_path:
            self.serialize(memory_data_path)

        self._post_process()

    def _post_process(self):
        # 产生切片
        self._generate_slice(self.slice_chunk_size)

        kv_bytes = self.blocks[self.router_layer_ids[0]].k.nbytes * len(self.router_layer_ids)
        if not self.vcache_in_cpu:
            kv_bytes *= 2

        kv_shape = self.blocks[self.router_layer_ids[0]].k.shape

        print(f"GPU {self.gpu_id} memory usage: {format_bytes(kv_bytes)} kv shape {kv_shape}")
        return kv_bytes, kv_shape

    def reset_memory(self):
        """Free per-document memory bank state without reloading the model.

        Drops blocks (per-layer KV tensors, the bulk of GPU/CPU memory), block
        descriptors, slice cache, and idx_to_doc. Keeps loaded model weights
        and template_prefix_kvcache (doc-independent). Caller must repopulate
        via generate_blocks(...) before serving queries again.
        """
        self.blocks = {layer: BlockData() for layer in self.router_layer_ids}
        self.block_desc = BlockDesc()
        self.k_slices = {}
        self.slice_desc = []
        if hasattr(self, 'idx_to_doc'):
            self.idx_to_doc = {}
        torch.cuda.empty_cache()

    def _prefill_template_vars(self):
        """
        计算 prefill 用的 template head tokens，与 PrefillStage1Worker._prepare_template 等价。
        结果缓存在 self._prefill_tmpl，避免重复 tokenize。
        """
        if hasattr(self, '_prefill_tmpl'):
            return self._prefill_tmpl

        pad_token = self.tokenizer.pad_token
        pad_token_id = self.tokenizer.pad_token_id
        prompt_template = self.generate_config.template["prompt"].replace("{prompt}", pad_token)
        prompt_template_inputs = self.tokenizer(prompt_template, add_special_tokens=False)
        prompt_template_input_ids = prompt_template_inputs["input_ids"]
        prompt_template_attention_mask = prompt_template_inputs["attention_mask"]
        pad_index = prompt_template_input_ids.index(pad_token_id)

        self._prefill_tmpl = {
            "pad_index": pad_index,
            "template_head_input_ids": prompt_template_input_ids[:pad_index],
            "template_head_attention_mask": prompt_template_attention_mask[:pad_index],
        }
        return self._prefill_tmpl

    def _prefill_block_direct(self, block: List[Document]) -> Dict:
        """
        用 self.model 直接 prefill 一批文档，不启动子进程。
        等价于 PrefillStage1Worker.inference，但在当前进程内执行。
        """
        tmpl = self._prefill_template_vars()
        pad_index = tmpl["pad_index"]
        template_head_input_ids = tmpl["template_head_input_ids"]
        template_head_attention_mask = tmpl["template_head_attention_mask"]
        pooling_kernel_size = self.memory_config.pooling_kernel_size
        template_id = -2

        # --- 构造输入（与 _prepare_block_inputs 等价）---
        doc_ids = [template_id] * len(template_head_input_ids)
        doc_input_ids = list(template_head_input_ids)
        doc_attention_mask = list(template_head_attention_mask)
        position_ids = list(range(pad_index))
        chunk_sizes = []

        for doc_idx, item in enumerate(block):
            doc_id, doc, pre_calculated_num_chunk = item.doc_id, item.doc, item.num_chunks
            _, doc_inputs = compose_input(doc, doc_id, self.tokenizer)
            length = len(doc_inputs["input_ids"])
            temp_doc_ids = [doc_idx + 1] * length
            temp_position_ids = list(range(length))
            chunk_size = (length + pooling_kernel_size - 1) // pooling_kernel_size
            assert chunk_size == pre_calculated_num_chunk, (
                f"pre calculated chunk {pre_calculated_num_chunk} got {chunk_size}, "
                f"doc str {len(doc)} id len {length}: [{doc_id}] <{doc}>"
            )
            chunk_sizes.append(chunk_size)
            doc_ids.extend(temp_doc_ids)
            doc_input_ids.extend(doc_inputs["input_ids"])
            doc_attention_mask.extend(doc_inputs["attention_mask"])
            position_ids.extend(temp_position_ids)

        input_ids_tensor = torch.LongTensor([doc_input_ids]).to(self.device)
        attention_mask_tensor = torch.LongTensor([doc_attention_mask]).to(self.device)
        doc_ids_tensor = torch.LongTensor([doc_ids]).to(self.device)
        position_ids_tensor = torch.LongTensor([position_ids]).to(self.device)

        # --- forward pass（与 _inference 等价）---
        past_key_values = CustomDynamicCacheOnCPU()
        for layer_idx in range(self.num_model_layers()):
            past_key_values.record_kwargs(layer_idx, {"stage": "prefill_stage1"})

        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
                position_ids=position_ids_tensor,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                output_docs_score=False,
                doc_ids=doc_ids_tensor,
            )

        torch.cuda.empty_cache()

        kv_meta = {
            "chunk_sizes": chunk_sizes,
            "past_key_values": outputs.past_key_values,
            "nr_docs": len(block),
            "doc_ids": [item.doc_id for item in block],
            "nr_chunks": [item.num_chunks for item in block],
        }
        return kv_meta

    def add_documents_incremental(self, new_docs: List[Document]) -> List[int]:
        """
        增量 prefill 新文档并追加 KV 到现有 tensors。

        流程：
        1. 用 self.model 直接 prefill 新文档（不启动子进程，避免重复载入模型导致 OOM）
        2. 将新 KV 在 dim[2] 追加到现有 block_data.k/v/rk
        3. 更新 block_desc（docs, doc_ids, pool_ids, offsets, lens）
        4. 调用 _post_process 重建 slices 和 k_slices

        Args:
            new_docs: 分配到本 GPU 的新文档列表，doc_id 已正确设置（全局唯一）

        Returns:
            新文档的全局 doc_id 列表
        """
        if not new_docs:
            return []

        # 收集新文档产生的 KV chunks（直接在当前进程用 self.model 推理）
        new_kv_per_layer: Dict[int, List[torch.Tensor]] = {layer: [] for layer in self.router_layer_ids}
        new_kv_v_per_layer: Dict[int, List[torch.Tensor]] = {layer: [] for layer in self.router_layer_ids}
        new_kv_rk_per_layer: Dict[int, List[torch.Tensor]] = {layer: [] for layer in self.router_layer_ids}
        new_pool_ids_parts: List[torch.Tensor] = []

        total_new_docs = len(new_docs)
        recv = 0
        pbar = tqdm(total=total_new_docs, desc=f"Worker-{self.gpu_id} incremental prefill", position=self.gpu_id)

        for block in PrefillStage1Worker.split_docs(new_docs, self.memory_config.block_size):
            meta = self._prefill_block_direct(block)
            recv += meta['nr_docs']
            pbar.update(meta['nr_docs'])

            past_key_values: CustomDynamicCacheOnCPU = meta["past_key_values"]

            for layer_idx in self.router_layer_ids:
                k, v = past_key_values.get_kvcache(layer_idx)
                rk = past_key_values.get_router_kcache(layer_idx)

                new_kv_per_layer[layer_idx].append(k)
                new_kv_v_per_layer[layer_idx].append(v)
                if rk is not None:
                    new_kv_rk_per_layer[layer_idx].append(rk)

                if layer_idx == self.router_layer_ids[0]:
                    new_pool_ids_parts.append(past_key_values.cache_kwargs[layer_idx]['pooled_doc_ids'])

            past_key_values.clear_kvcache()

        pbar.close()

        # --- 1. 合并新 pool_ids（局部 id，从1起始，每批独立）---
        # cumulative_concat 使得跨批次的 pool_ids 连续递增
        new_pool_ids_local = cumulative_concat(new_pool_ids_parts)  # CPU tensor

        # --- 2. 将新 pool_ids 偏移到全局空间 ---
        # 现有 pool_ids 最大值就是现有文档总数（local index）
        existing_max_pool_id = self.block_desc.pool_ids[-1].item() if (
            self.block_desc.pool_ids is not None and self.block_desc.pool_ids.numel() > 0
        ) else 0
        new_pool_ids_global_cpu = new_pool_ids_local + existing_max_pool_id
        new_pool_ids_global = new_pool_ids_global_cpu.to(self.device)

        # --- 3. 拼接新 KV 到现有 tensors ---
        for layer_idx in self.router_layer_ids:
            block_data = self.blocks[layer_idx]

            # 合并本次新 KV（跨 batch）
            new_k = torch.cat(new_kv_per_layer[layer_idx], dim=2)   # [1, H, new_chunks, D]
            new_v = torch.cat(new_kv_v_per_layer[layer_idx], dim=2)

            # 追加到现有 tensors（dim=2 是 chunk 维度）
            block_data.k = torch.cat([block_data.k, new_k.to(block_data.k.device)], dim=2)
            block_data.v = torch.cat([block_data.v, new_v.to(block_data.v.device).to(block_data.v.dtype)], dim=2)

            if block_data.rk is not None and new_kv_rk_per_layer[layer_idx]:
                new_rk = torch.cat(new_kv_rk_per_layer[layer_idx], dim=2)
                block_data.rk = torch.cat([block_data.rk, new_rk.to(block_data.rk.device)], dim=2)

        # --- 4. 更新 block_desc ---
        desc = self.block_desc

        # 4a. 追加 docs 列表
        desc.docs = desc.docs + new_docs

        # 4b. 追加 doc_ids（全局 doc_id 查找表，位置 0 是哨兵 0）
        new_global_doc_ids = torch.LongTensor([doc.doc_id for doc in new_docs])
        desc.doc_ids_cpu = torch.cat([desc.doc_ids_cpu, new_global_doc_ids])
        desc.doc_ids = desc.doc_ids_cpu.to(self.device)

        # 4c. 合并 pool_ids
        desc.pool_ids = torch.cat([desc.pool_ids, new_pool_ids_global])
        desc.pool_ids_cpu = desc.pool_ids.cpu()

        # 4d. 更新 doc_offsets_cpu 和 doc_lens_cpu
        # 这两个数组以 local pool_id（即文档在 bucket 内的顺序）为索引
        # 新文档追加到末尾
        new_lens = torch.LongTensor([doc.num_chunks for doc in new_docs])
        # doc_offsets_cpu[i] = 第 i 个文档起始 chunk 偏移（pool_id 从 1 起，0 是哨兵）
        existing_total_chunks = desc.doc_offsets_cpu[-1].item() + desc.doc_lens_cpu[-1].item()
        new_offsets = existing_total_chunks + torch.cumsum(
            torch.cat([torch.tensor([0]), new_lens[:-1]]), dim=0
        )
        desc.doc_offsets_cpu = torch.cat([desc.doc_offsets_cpu, new_offsets])
        desc.doc_lens_cpu = torch.cat([desc.doc_lens_cpu, new_lens])

        # --- 5. 重建 slices 和 k_slices ---
        self._post_process()

        return [doc.doc_id for doc in new_docs]

    @staticmethod
    def map_tensor_to_group_ids(a: torch.Tensor) -> torch.Tensor:
        """
        根据相邻元素是否相同，将输入 Tensor a 的值映射到递增的组 ID Tensor b。

        a[i] == a[i-1] => b[i] = b[i-1]
        a[i] != a[i-1] => b[i] = b[i-1] + 1
        b[0] 始终为 1。

        Args:
            a (torch.Tensor): 待处理的一维 int Tensor，必须在 GPU 上。

        Returns:
            torch.Tensor: 生成的组 ID Tensor b，在 GPU 上。
        """
        if a.ndim != 1:
            raise ValueError("输入 Tensor a 必须是一维的。")

        # 1. 检查相邻元素是否不同 (a[i] != a[i-1])
        # torch.diff(a) 计算相邻元素之间的差值 a[i] - a[i-1]
        # 差值不为 0 (!= 0) 即表示 a[i] != a[i-1]
        # 结果是一个布尔 Tensor，长度比 a 短 1，代表从 a[1] 开始的相邻比较
        diff_mask = torch.diff(a) != 0  # [L-1]

        # 2. 将布尔值转换为整数 (0 或 1)
        # True -> 1 (需要增加 ID)，False -> 0 (保持 ID)
        # torch.diff 的结果长度是 L-1，我们只关心从 a[1] 开始的递增
        id_increments = diff_mask.int()  # [L-1]

        # 3. 累计增量
        # torch.cumsum 对增量进行累加：
        # 结果: [0, 0+1, 0+1+0, ...]
        # 长度仍然是 L-1
        group_indices_offset = torch.cumsum(id_increments, dim=0)  # [L-1]

        # 4. 构造最终的 Tensor b
        # b[0] 始终是 1
        # 对于 i > 0，b[i] = group_indices_offset[i-1] + 1
        
        # 在 group_indices_offset 前面添加一个 0 作为 a[0] 的基准偏移
        # [0, group_indices_offset[0], group_indices_offset[1], ...]
        # 结果长度为 L
        b = torch.cat((
            torch.tensor([0], device=a.device, dtype=a.dtype), 
            group_indices_offset
        )) + 1
        return b
    def _generate_slice(self, chunks):
        nr_doc_slices = []  # 每个 block slice 所含的文档数量
        nr_chunk_slices = [] # 每个 block slice 所含的 chunk 数量
        chunk_slices_offset = [] # 每个 block slice 所含的 chunk 位置
        chunk_slice_position: Tuple[int, int] = [] # 每个 chunk slice 的起始和结束位置

        current_chunks = 0
        current_docs = 0
        chunk_offset = 0
        for doc in self.block_desc.docs:
            current_chunks += doc.num_chunks
            current_docs += 1
            chunk_offset += doc.num_chunks
            
            if current_chunks >= chunks:
                nr_doc_slices.append(current_docs)
                nr_chunk_slices.append(current_chunks)
                chunk_slices_offset.append(chunk_offset)
                current_chunks = 0
                current_docs = 0

        if current_chunks > 0:
            if current_chunks < 1024 and len(nr_doc_slices) > 0:
                # too small, append to last
                nr_doc_slices[-1] += current_docs
                nr_chunk_slices[-1] += current_chunks
                chunk_slices_offset[-1] = chunk_offset
            else:
                nr_doc_slices.append(current_docs)
                nr_chunk_slices.append(current_chunks)
                chunk_slices_offset.append(chunk_offset)

        starts = [0] + chunk_slices_offset[:-1]
        chunk_slice_position = [(start, end) for start, end in zip(starts, chunk_slices_offset)]

        self.k_slices = {}
        self.slice_desc = []
        for slice_idx, (start, end) in enumerate(chunk_slice_position):
            desc = SliceDesc(nr_docs=nr_doc_slices[slice_idx],
                             nr_chunks=nr_chunk_slices[slice_idx],
                             global_doc_ids=self.block_desc.pool_ids[start:end],
                             local_doc_ids=Memory.map_tensor_to_group_ids(self.block_desc.pool_ids[start:end]))
            desc.local_doc_ids_0 = (desc.local_doc_ids - 1).view(1, -1)
            unique_indices = torch.cat([torch.tensor([0], device=self.device), torch.where(desc.local_doc_ids[1:] != desc.local_doc_ids[:-1])[0] + 1])
            desc.original_doc_ids = desc.global_doc_ids[unique_indices].view(1, -1)
            
            self.slice_desc.append(desc)
        
        for layer_idx in self.router_layer_ids:
            rk = self.blocks[layer_idx].get_router_k()
            slice = [rk[:,:,start:end].transpose(-1, -2).unsqueeze(2) for start, end in chunk_slice_position]
            self.k_slices[layer_idx] = slice

    def get_max_local_pool_doc_id(self) -> List[int]:
        return self.block_desc.pool_ids[-1].item()
        
    def get_template_prefix_kvcaches(self, layer_idx: int):
        return self.template_prefix_kvcache[layer_idx]

    def prefill_stage2(self, layer_idx: int, query_states: torch.Tensor, query_mask: torch.Tensor):
        # query_mask: [bsz, 1, 1, seqlen, 1]
        # query_states: [bsz, n_kv_head, n_kv_groups, seqlen, hdim]

        bsz, num_kv_heads, num_kv_groups, seq_len, head_dim = query_states.shape
        min_val = torch.finfo(query_states.dtype).min
        routing_mask = ~query_mask
        query_mask = query_mask.squeeze(2)  # [bsz, 1, seqlen, 1]
        q_lens = None

        # 初始化分数表
        # 严重注意：请确认 nr_docs 是否真的等于 max_doc_id + 1。
        # 如果 pool_ids 是全局唯一的（例如加了 bias 100000），这里的 total_docs 必须能容纳最大的 ID。
        total_docs = self.block_desc.nr_docs + 1 
        global_doc_scores = torch.full((bsz, total_docs), min_val, dtype=query_states.dtype, device=self.device)

        # --- Phase 1: 打分 (移除 Query Batch Loop) ---
        
        # 针对 A100，如果 bsz < 256，直接全量计算收益最高
        # 我们直接遍历 slice，不再对 query 进行切片
        
        for slice_desc, k_slice_t in zip(self.slice_desc, self.k_slices[layer_idx]):
            # k_slice_t: [1, num_kv_heads, 1, hdim, chunk] (GPU)
            
            # 1. Matmul (Compute Bound)
            # [bsz, num_kv_heads, groups, seqlen, hdim] @ [1, num_kv_heads, 1, hdim, chunk] 
            # -> [bsz, num_kv_heads, groups, seqlen, chunk]
            attn_scores = torch.matmul(query_states, k_slice_t)
            
            # 2. Mask & Reduce (Memory Bound)
            attn_scores.masked_fill_(routing_mask, min_val)
            # Flatten & Reduce heads/seqlen
            # flatten(1,2): [bsz, num_kv_heads*groups, seqlen, chunk]
            # max(dim=2)[0]: [bsz, num_kv_heads*groups, chunk]
            # max(dim=1)[0]: [bsz, chunk]
            # max_scores_per_chunk = attn_scores.flatten(1, 2).max(dim=2)[0].max(dim=1)[0] # [bsz, chunk]
            max_scores_per_chunk = attn_scores.flatten(1, 2)

            # [bsz, num_kv_heads*groups, chunk]
            if self.head_reduce_method == "max":
                max_scores_per_chunk = max_scores_per_chunk.max(dim=1).values
            elif self.head_reduce_method == "mean":
                max_scores_per_chunk = max_scores_per_chunk.mean(dim=1)
            else:
                raise NotImplementedError(f"Unsupported head reduce method: {self.head_reduce_method}")
            
            if self.query_reduce_method == "max":
                max_scores_per_chunk = max_scores_per_chunk.max(dim=1).values
            elif self.query_reduce_method == "mean":
                # 1. 调整 mask 形状以匹配 max_scores_per_chunk
                # [bsz, 1, seqlen, 1] -> [bsz, seqlen, 1]
                mask = query_mask.squeeze(1) 

                # 2. 在求和前，将无效位置（min_val）归零
                # 使用 torch.where 比乘法更安全，且能有效处理极小的 min_val
                masked_sum = torch.where(mask, max_scores_per_chunk, torch.zeros_like(max_scores_per_chunk)).sum(dim=1)

                # 3. 计算每个 sequence 中的有效 token 数量
                # 防止除以 0，使用 clamp 或加一个极小值 eps
                valid_counts = mask.sum(dim=1).clamp(min=1.0) 

                # 4. 得到最终均值 [bsz, chunk]
                max_scores_per_chunk = masked_sum / valid_counts
            elif self.query_reduce_method == "last":
                if q_lens is None:
                    q_lens = query_mask.squeeze(3).squeeze(1).sum(dim=1).long()
                
                # 边界保护: 如果 length 为 0 (异常情况), clamp 到 0
                last_indices = (q_lens - 1).clamp(min=0)
                
                # 3. 调整形状用于 gather
                # [B, 1, C]
                gather_idx = last_indices.view(bsz, 1, 1).expand(-1, 1, slice_desc.nr_chunks)
                
                # 4. 提取
                max_scores_per_chunk = max_scores_per_chunk.gather(1, gather_idx).squeeze(1)
            else:
                raise NotImplementedError(f"Unsupported query reduce method: {self.query_reduce_method}")

            # 3. Local Scatter (Chunk -> Doc)
            # slice_desc.local_doc_ids_0 已经在 GPU 上预计算好了
            # [1, chunk] -> [bsz, chunk]
            scatter_indices = slice_desc.local_doc_ids_0.expand(bsz, -1)
            
            local_doc_scores = torch.full((bsz, slice_desc.nr_docs), -float('inf'), device=self.device, dtype=query_states.dtype)
            if self.chunk_reduce_method == "max":
                local_doc_scores.scatter_reduce_(
                    dim=1, index=scatter_indices, src=max_scores_per_chunk, 
                    reduce="amax", include_self=True
                )
            
            elif self.chunk_reduce_method == "mean":
                doc_sums = torch.zeros_like(local_doc_scores)
                doc_sums = doc_sums.scatter_reduce(
                    dim=1, index=scatter_indices, src=max_scores_per_chunk, 
                    reduce="sum", include_self=False
                )
                doc_counts = torch.zeros_like(local_doc_scores)
                ones = torch.ones_like(max_scores_per_chunk)
                doc_counts = doc_counts.scatter_reduce(
                    dim=1, index=scatter_indices, src=ones, 
                    reduce="sum", include_self=False
                )
                doc_counts = doc_counts.clamp(min=1.0)
                mean_scores = doc_sums / doc_counts
                # TODO: 能不能不要重新产生一个local_doc_scores？
                local_doc_scores = torch.where(doc_counts > 0, mean_scores, local_doc_scores)

                del doc_sums, doc_counts, mean_scores
            else:
                raise ValueError(f"Invalid chunk reduction method: {self.chunk_reduce_method}")
            
            # 4. Global Scatter (Local Doc -> Global Doc)
            global_indices = slice_desc.original_doc_ids.expand(bsz, -1)
            global_doc_scores.scatter_reduce_(
                dim=1, index=global_indices, src=local_doc_scores, 
                reduce="amax", include_self=True
            )
            del attn_scores, max_scores_per_chunk, local_doc_scores

        # --- Phase 2: Top-K 选择 (使用 topk 替代 sort) ---
        
        # 优化点：topk 比 sort 快得多
        final_scores, batch_selected_doc_ids = torch.topk(global_doc_scores, k=min(self.model_config.doc_top_k, global_doc_scores.shape[1]), dim=1) # [bsz, k]
        batch_selected_global_doc_ids = self.block_desc.create_global_doc_ids(batch_selected_doc_ids)
        return final_scores, batch_selected_doc_ids, batch_selected_global_doc_ids

    def prefill_stage2_1(self, layer_idx: int, query_states: torch.Tensor, query_mask: torch.Tensor):
        # ... [Timer Start] ...
        
        # 优化点：Mask可以稍后处理，先让 Query 进计算
        
        bsz, num_heads, seq_len, head_dim = query_states.shape
        num_kv_groups = self.num_key_value_groups 
        num_kv_heads = num_heads // num_kv_groups
        min_val = torch.finfo(query_states.dtype).min
        routing_mask = ~query_mask
        q_lens = None
        
        # 1. View 重塑 (Zero-copy)
        # TODO: scaling 计算
        scaling = self.scaling
        query_states = query_states.view(bsz, num_kv_heads, num_kv_groups, seq_len, head_dim) * scaling

        # 初始化分数表
        # 严重注意：请确认 nr_docs 是否真的等于 max_doc_id + 1。
        # 如果 pool_ids 是全局唯一的（例如加了 bias 100000），这里的 total_docs 必须能容纳最大的 ID。
        total_docs = self.block_desc.nr_docs + 1 
        global_doc_scores = torch.full((bsz, total_docs), min_val, dtype=query_states.dtype, device=self.device)

        # --- Phase 1: 打分 (移除 Query Batch Loop) ---
        
        # 针对 A100，如果 bsz < 256，直接全量计算收益最高
        # 我们直接遍历 slice，不再对 query 进行切片
        
        for slice_desc, k_slice_t in zip(self.slice_desc, self.k_slices[layer_idx]):
            # k_slice_t: [1, nhead, 1, hdim, chunk] (GPU)
            
            # 1. Matmul (Compute Bound)
            # [bsz, nhead, groups, seqlen, hdim] @ [1, nhead, 1, hdim, chunk] 
            # -> [bsz, nhead, groups, seqlen, chunk]
            attn_scores = torch.matmul(query_states, k_slice_t)
            
            # 2. Mask & Reduce (Memory Bound)
            attn_scores.masked_fill_(routing_mask, min_val)
            # Flatten & Reduce heads/seqlen
            # flatten(1,2): [bsz, nhead*groups, seqlen, chunk]
            # max(dim=2)[0]: [bsz, nhead*groups, chunk]
            # max(dim=1)[0]: [bsz, chunk]
            # max_scores_per_chunk = attn_scores.flatten(1, 2).max(dim=2)[0].max(dim=1)[0] # [bsz, chunk]
            max_scores_per_chunk = attn_scores.flatten(1, 2)

            # [bsz, nhead*groups, chunk]
            if self.head_reduce_method == "max":
                max_scores_per_chunk = max_scores_per_chunk.max(dim=1).values
            elif self.head_reduce_method == "mean":
                max_scores_per_chunk = max_scores_per_chunk.mean(dim=1)
            else:
                raise NotImplementedError(f"Unsupported head reduce method: {self.head_reduce_method}")
            
            if self.query_reduce_method == "max":
                max_scores_per_chunk = max_scores_per_chunk.max(dim=1).values
            elif self.query_reduce_method == "mean":
                # 1. 调整 mask 形状以匹配 max_scores_per_chunk
                # [bsz, 1, seqlen, 1] -> [bsz, seqlen, 1]
                mask = query_mask.squeeze(1) 

                # 2. 在求和前，将无效位置（min_val）归零
                # 使用 torch.where 比乘法更安全，且能有效处理极小的 min_val
                masked_sum = torch.where(mask, max_scores_per_chunk, torch.zeros_like(max_scores_per_chunk)).sum(dim=1)

                # 3. 计算每个 sequence 中的有效 token 数量
                # 防止除以 0，使用 clamp 或加一个极小值 eps
                valid_counts = mask.sum(dim=1).clamp(min=1.0) 

                # 4. 得到最终均值 [bsz, chunk]
                max_scores_per_chunk = masked_sum / valid_counts
            elif self.query_reduce_method == "last":
                if q_lens is None:
                    q_lens = query_mask.squeeze(3).squeeze(1).sum(dim=1).long()
                
                # 边界保护: 如果 length 为 0 (异常情况), clamp 到 0
                last_indices = (q_lens - 1).clamp(min=0)
                
                # 3. 调整形状用于 gather
                # [B, 1, C]
                gather_idx = last_indices.view(bsz, 1, 1).expand(-1, 1, slice_desc.nr_chunks)
                
                # 4. 提取
                max_scores_per_chunk = max_scores_per_chunk.gather(1, gather_idx).squeeze(1)
            else:
                raise NotImplementedError(f"Unsupported query reduce method: {self.query_reduce_method}")

            # 3. Local Scatter (Chunk -> Doc)
            # slice_desc.local_doc_ids_0 已经在 GPU 上预计算好了
            # [1, chunk] -> [bsz, chunk]
            scatter_indices = slice_desc.local_doc_ids_0.expand(bsz, -1)
            
            local_doc_scores = torch.full((bsz, slice_desc.nr_docs), -float('inf'), device=self.device, dtype=query_states.dtype)
            if self.chunk_reduce_method == "max":
                local_doc_scores.scatter_reduce_(
                    dim=1, index=scatter_indices, src=max_scores_per_chunk, 
                    reduce="amax", include_self=True
                )
            
            elif self.chunk_reduce_method == "mean":
                doc_sums = torch.zeros_like(local_doc_scores)
                doc_sums = doc_sums.scatter_reduce(
                    dim=1, index=scatter_indices, src=max_scores_per_chunk, 
                    reduce="sum", include_self=False
                )
                doc_counts = torch.zeros_like(local_doc_scores)
                ones = torch.ones_like(max_scores_per_chunk)
                doc_counts = doc_counts.scatter_reduce(
                    dim=1, index=scatter_indices, src=ones, 
                    reduce="sum", include_self=False
                )
                doc_counts = doc_counts.clamp(min=1.0)
                mean_scores = doc_sums / doc_counts
                # TODO: 能不能不要重新产生一个local_doc_scores？
                local_doc_scores = torch.where(doc_counts > 0, mean_scores, local_doc_scores)

                del doc_sums, doc_counts, mean_scores
            else:
                raise ValueError(f"Invalid chunk reduction method: {self.chunk_reduce_method}")
            
            # 4. Global Scatter (Local Doc -> Global Doc)
            global_indices = slice_desc.original_doc_ids.expand(bsz, -1)
            global_doc_scores.scatter_reduce_(
                dim=1, index=global_indices, src=local_doc_scores, 
                reduce="amax", include_self=True
            )
            del attn_scores, max_scores_per_chunk, local_doc_scores

        # --- Phase 2: Top-K 选择 (使用 topk 替代 sort) ---
        
        # 优化点：topk 比 sort 快得多
        final_scores, batch_selected_doc_ids = torch.topk(global_doc_scores, k=self.model_config.doc_top_k, dim=1) # [bsz, k]
        
        required_doc_mask = torch.zeros(total_docs, dtype=torch.bool, device=self.device)
        required_doc_mask.scatter_(0, batch_selected_doc_ids.flatten(), True)
        
        if not required_doc_mask.any():
             return None, None, None, None

        # --- Phase 3: 异步并行提取 (Async Retrieval) ---
        
        block = self.blocks[layer_idx]
        pooled_doc_ids = self.block_desc.pool_ids # [num_chunks]
        
        # 计算 Chunk Mask (GPU)
        required_chunk_mask = required_doc_mask[pooled_doc_ids] # [num_chunks] Bool

        # 确保先处理 K
        k_selected = block.k[:, :, required_chunk_mask, :]
        pooled_doc_ids_selected = self.block_desc.create_global_doc_ids(pooled_doc_ids[required_chunk_mask])

        cpu_indices = torch.where(required_chunk_mask)[0].to("cpu")
        if rk_selected.is_cpu:
            rk_selected = block.rk[:, :, cpu_indices, :]
            rk_selected = rk_selected.to(self.device, non_blocking=True)
        else:
            rk_selected = block.rk[:, :, required_chunk_mask, :]

        v_selected = block.v[:, :, cpu_indices, :].to(self.device, non_blocking=True)

        # 由于 K 和 IDs 用了 non_blocking，我们需要在使用它们之前确保传输完成
        # 这里的 synchronize 确保 k_retrieved 和 pooled_doc_ids_retrieved 数据已就绪
        torch.cuda.current_stream().synchronize()

        return final_scores, k_selected, rk_selected, v_selected, pooled_doc_ids_selected

    def gpu_select(self, scores, k, rk, v, pooled_doc_ids, total_docs):
        final_scores, batch_selected_doc_ids = torch.topk(scores, k=self.model_config.doc_top_k, dim=1) # [bsz, k]
        
        required_doc_mask = torch.zeros(total_docs, dtype=torch.bool, device=self.device)
        required_doc_mask.scatter_(0, batch_selected_doc_ids.flatten(), True)
        
        # 计算 Chunk Mask (GPU)
        required_chunk_mask = required_doc_mask[pooled_doc_ids] # [num_chunks] Bool

        # 确保先处理 K
        k_selected = k[:, :, required_chunk_mask, :]
        pooled_doc_ids_selected = pooled_doc_ids[required_chunk_mask]
        rk_selected = rk[:, :, required_chunk_mask, :]
        v_selected = v[:, :, required_chunk_mask, :]

        return final_scores, k_selected, rk_selected, v_selected, pooled_doc_ids_selected


class MSAService(Memory, MemoryClientBase):
    def __init__(self,
                 gpu_id: int,
                 generate_config: GenerateConfig,
                 model_config: ModelConfig,
                 memory_config: MemoryConfig
                 ):
        self.world_size = generate_config.world

        # 初始化 NCCL
        # 注意：这里假设是在单机多卡环境下运行，使用 localhost
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        
        # 初始化进程组
        # timeout 设置稍长一点，防止初始化时某些进程慢导致超时
        dist.init_process_group(
            "nccl", 
            rank=gpu_id, 
            world_size=self.world_size
        )
        print(f"[GPU {gpu_id}] NCCL Initialized successfully.")

        super().__init__(gpu_id, generate_config, model_config, memory_config)

        self.generate_kwarg = self._create_generate_args()

    def _create_generate_args(self):
        generate_kwarg = {
            "do_sample": True,
            "top_p": self.generate_config.top_p,
            "temperature": self.generate_config.temperature,
            "max_new_tokens": self.generate_config.max_generate_tokens,
        }
        if self.generate_config.temperature == 0.0:
            generate_kwarg["do_sample"] = False
            generate_kwarg["temperature"] = None
            generate_kwarg["top_p"] = None
            generate_kwarg["top_k"] = None
        return generate_kwarg


    def setup_memory_client(self, model):
        def _setup_module(module):
            if hasattr(module, 'set_memory_client') and callable(module.set_memory_client):
                try:
                    module.set_memory_client(self)
                except Exception as e:
                    print(f"✗ 调用 {module.__class__.__name__} 的 set_memory_client() 失败: {e}")
        
        # 递归遍历所有模块
        for _, module in model.named_modules():
            _setup_module(module)

        
    def load_model(self):
        """加载模型和tokenizer"""
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_path)

        # 加载模型
        self.model = MSAForCausalLM.from_pretrained(
            self.model_config.model_path,
            use_cache=True,
            attn_implementation="flash_attention_2",
            torch_dtype="auto",
            device_map=self.device,
        )

        self.model.eval()
        self.setup_memory_client(self.model)

    
    def _gather_querys(self, query_states: torch.Tensor, query_mask: torch.Tensor):
        """处理变长 seqlen：填充到统一长度并 all_gather"""
        bsz, nhead, seqlen, hdim = query_states.shape
        num_kv_groups = self.num_key_value_groups 
        num_kv_heads = nhead // num_kv_groups
        world_size = self.world_size
        device = self.device
        if self.scaling > 0:
            query_states = query_states * self.scaling

        # --- Step 1: 准备全量 Query (分布式 All-Gather) ---
        m_for_search = query_mask.view(bsz, 1, 1, seqlen, 1)
        q_for_search = query_states.view(bsz, num_kv_heads, num_kv_groups, seqlen, hdim)

        # --- Step 2: 处理变长 Seqlen (隐式同步处理) ---
        local_seqlen = torch.tensor([seqlen], device=device, dtype=torch.long)
        all_seqlens = [torch.empty(1, device=device, dtype=torch.long) for _ in range(world_size)]
        dist.all_gather(all_seqlens, local_seqlen)
        
        # 此处 s.item() 确实会触发同步，但这是为了在 CPU 端分配 all_queries 的 List
        seqlens = [s.item() for s in all_seqlens]
        max_seqlen = max(seqlens)

        # --- Step 3: 预分配连续内存进行 All-Gather (解决 Contiguous 问题) ---
        # 我们预先分配一个大的连续 Buffer，然后把 local 数据 copy 进去
        # 这样 gather 出来的数据天生就是连续的，不需要调 .contiguous()
        def get_padded_gather_list(tensor, target_seqlen):
            # 构造 padded 形状
            padded_shape = list(tensor.shape)
            padded_shape[3] = target_seqlen # 修改 seqlen 维度
            
            # 分配全量列表
            gather_list = [torch.zeros(padded_shape, dtype=tensor.dtype, device=device) for _ in range(world_size)]
            
            # 将本地数据 copy 到 gather_list 中对应的位置
            # 这里 slice 操作 + copy_ 比先 pad 再 gather 更节省临时显存
            gather_list[self.gpu_id][:, :, :, :seqlen, :].copy_(tensor)
            dist.all_gather(gather_list, gather_list[self.gpu_id])
            return gather_list

        all_queries = get_padded_gather_list(q_for_search, max_seqlen)
        all_masks = get_padded_gather_list(m_for_search, max_seqlen)

        return all_queries, all_masks, seqlens

    def doc_query(self, query_states: torch.Tensor, query_mask: torch.Tensor, layer_idx: int):
        """
        集成版：检索 + 跨卡提取 + 查表加速 + 结果归集
        返回: 
            final_k, final_v: [H, Total_Selected_C, D] 格式，直接用于 Flash Attn 散射
            final_scores: [B, TopK] 文档检索分数
            num_selected_chunks_per_sample: [B] 每个 Batch 选中的 chunk 总数
        """
        world_size = self.world_size
        bsz, nhead, seqlen, hdim = query_states.shape
        num_kv_groups = self.num_key_value_groups 
        num_kv_heads = nhead // num_kv_groups
        device = self.device
        dtype = query_states.dtype
        top_k = self.model_config.doc_top_k

        all_queries, all_masks, seqlens = self._gather_querys(query_states, query_mask)

        # --- Step 2: 循环处理各个 Rank 的请求并本地打分 ---
        scores_list, combined_ids_list = [], []
        for target_rank in range(world_size):
            seqlen = seqlens[target_rank]
            q_batch = all_queries[target_rank][:, :, :seqlen, :]
            m_batch = all_masks[target_rank][:, :, :seqlen, :]
            # q_batch 来自 target_rank，我们需要在当前 GPU 上寻找最佳匹配
            l_scores, l_ids, g_ids = self.prefill_stage2(layer_idx, q_batch, m_batch)
            scores_list.append(l_scores)     # [bsz, top_k]
            # 将 local_ids 和 global_ids 打包在最后一个维度，减少一次 all-to-all
            combined_ids_list.append(torch.stack([l_ids, g_ids], dim=-1))

        # --- Step 3: 元数据交换 (All-to-All) ---
        # 发送：我为各 Rank 算的 Top-K；接收：各 Rank 为“我”算的 Top-K
        send_scores = torch.stack(scores_list, dim=0) # [world_size, bsz, top_k]
        send_combined_ids = torch.stack(combined_ids_list, dim=0) # [world_size, bsz, top_k, 2]
        
        recv_scores = torch.empty_like(send_scores)
        recv_combined_ids = torch.empty_like(send_combined_ids)
        
        dist.all_to_all_single(recv_scores, send_scores)
        dist.all_to_all_single(recv_combined_ids, send_combined_ids)
        
        # --- Step 4: 全局 Top-K 决策 ---
        # shapes: [world_size, bsz, top_k] -> [bsz, world_size * top_k]
        candidate_scores = recv_scores.transpose(0, 1).reshape(bsz, -1)
        candidate_local_ids = recv_combined_ids[..., 0].transpose(0, 1).reshape(bsz, -1)
        candidate_global_ids = recv_combined_ids[..., 1].transpose(0, 1).reshape(bsz, -1)
        
        origin_ranks = torch.arange(world_size, device=device).view(world_size, 1, 1).expand(-1, bsz, top_k) # [world_size, bsz, top_k]
        origin_ranks = origin_ranks.transpose(0, 1).reshape(bsz, -1) # [bsz, world_size * top_k]

        # [bsz, top_k]
        final_scores, final_indices = torch.topk(candidate_scores, k=top_k, dim=1)
        winner_local_ids = candidate_local_ids.gather(1, final_indices)
        winner_global_ids = candidate_global_ids.gather(1, final_indices)
        winner_ranks = origin_ranks.gather(1, final_indices)

        # 屏蔽掉无效分数
        valid_mask = final_scores > -1e9
        winner_local_ids = winner_local_ids.masked_fill(~valid_mask, -1)

        # --- Step 5: 任务下发 (Request) ---
        # 告诉其他 Rank：我最终选了你哪几个 local_id
        task_mask = torch.arange(world_size, device=device).view(world_size, 1, 1)
        tasks_to_send = torch.where(winner_ranks == task_mask, winner_local_ids, -1)
        remote_tasks = torch.empty_like(tasks_to_send)
        dist.all_to_all_single(remote_tasks, tasks_to_send)
        remote_tasks_cpu = remote_tasks.to("cpu")
        
        # --- Step 6: 提取 KV (核心：针对百万级数据的查表加速) ---
        
        block = self.blocks[layer_idx]
        desc = self.block_desc

        # 1. 预处理：收集每个 Rank 的请求元数据 (在 CPU 上完成，极快)
        all_rank_local_ids = []
        chunks_per_rank = []  # 记录每个 Rank 分得的 chunk 总数，用于后续拆分
        all_rank_u_lens = []  # 记录每个 Rank 选中的每个 doc 的长度

        for r in range(world_size):
            requested_ids = remote_tasks_cpu[r]
            # 获取该 Rank 选中的唯一 local_doc_ids
            unique_req_ids = requested_ids[requested_ids != -1].unique()
            all_rank_local_ids.append(unique_req_ids)
            
            if unique_req_ids.numel() > 0:
                u_lens = desc.doc_lens_cpu[unique_req_ids]
                all_rank_u_lens.append(u_lens)
                chunks_per_rank.append(u_lens.sum().item())
            else:
                all_rank_u_lens.append(torch.empty(0, dtype=torch.long))
                chunks_per_rank.append(0)

        # 2. 构造全局批次索引 (One-shot Index Generation)
        flat_unique_ids = torch.cat(all_rank_local_ids)
        total_chunks_all_ranks = sum(chunks_per_rank)

        # 直接在 fetch 过程中完成 K 和 V 的合并
        # 预分配用于 all-to-all 的连续显存
        # total_chunks 是当前 Rank 需要发给所有其他 Rank 的 chunk 总和
        send_kv_combined = torch.empty((total_chunks_all_ranks, 2, num_kv_heads, hdim), device=device, dtype=dtype)

        if total_chunks_all_ranks > 0:
            # 批量获取所有选定文档的偏移和长度
            batch_u_offsets = desc.doc_offsets_cpu[flat_unique_ids]
            batch_u_lens = torch.cat(all_rank_u_lens)
            
            # 向量化生成全局索引列表
            base_indices = torch.repeat_interleave(batch_u_offsets, batch_u_lens)
            inner_seq = torch.arange(total_chunks_all_ranks) - torch.repeat_interleave(
                torch.cumsum(batch_u_lens, dim=0) - batch_u_lens, batch_u_lens
            )
            all_cpu_indices = base_indices + inner_seq
            all_gpu_indices = all_cpu_indices.to(device, non_blocking=True)
            
            # 3. 一次性提取并搬运 (核心性能提升点)
            # 由于 block.k/v 是 pinned，这一步是满速 DMA 异步传输
            # [1, H, C_all, D]
            def fetch(tensor, slot_idx):
                # tensor: [1, H, C, D] -> index_select -> [1, H, Sel_C, D]
                if tensor.is_cpu:
                    selected = tensor.index_select(2, all_cpu_indices).to(device, non_blocking=True)
                else:
                    selected = tensor.index_select(2, all_gpu_indices)
                send_kv_combined[:, slot_idx, :, :].copy_(selected.squeeze(0).transpose(0, 1), non_blocking=True)

            fetch(block.k, 0)
            fetch(block.v, 1)
            
            # 批量处理全局 Doc ID 映射
            all_g_ids_rep = desc.doc_ids_cpu[flat_unique_ids].repeat_interleave(batch_u_lens).to(device, non_blocking=True)
        else:
            # 全量空处理
            all_g_ids_rep = torch.empty(0, device=device, dtype=torch.long)

        send_num_chunks = torch.tensor(chunks_per_rank, device=device)

        # --- Step 7: 变长数据交换 (All-to-All) ---
        recv_num_chunks = torch.zeros_like(send_num_chunks)
        dist.all_to_all_single(recv_num_chunks, send_num_chunks)
        torch.cuda.current_stream().synchronize() # 确保 fetch(CPU) 已完成同步
        
        total_recv = recv_num_chunks.sum().item()
        recv_kv_flat = torch.empty(total_recv, 2, num_kv_heads, hdim, dtype=dtype, device=device)
        recv_g_ids_flat = torch.empty(total_recv, dtype=torch.long, device=device)
        
        in_splits = send_num_chunks.tolist()
        out_splits = recv_num_chunks.tolist()

        dist.all_to_all_single(recv_kv_flat, send_kv_combined, out_splits, in_splits)
        dist.all_to_all_single(recv_g_ids_flat, all_g_ids_rep, out_splits, in_splits)
        torch.cuda.current_stream().synchronize() 

        # --- Step 8: 后期处理集成 (按 Batch 重排归集) ---
        # 构造匹配矩阵 [B, TopK, Total_Recv_Chunks]
        # 利用广播找到每个 Batch Item 选中的 chunk
        # TODO: 这里可能会爆显存，后续考虑分块处理
        match_matrix = (winner_global_ids.unsqueeze(-1) == recv_g_ids_flat.unsqueeze(0).unsqueeze(0))
        mask_per_batch = match_matrix.any(dim=1) # [B, Total_Recv_Chunks]

        # 假设 recv_k_flat 形状为 [C, 1, H, D]
        # 1. 调整维度，使 B 和 C 排在最前面，方便 mask 索引
        # unsqueeze(0) -> [1, C, 1, H, D]
        # expand(bsz, ...) -> [bsz, C, 1, H, D]
        candidate_k_exp = recv_kv_flat[:,0:1, :, :].unsqueeze(0).expand(bsz, -1, -1, -1, -1) 
        candidate_v_exp = recv_kv_flat[:,1:2, :, :].unsqueeze(0).expand(bsz, -1, -1, -1, -1) 

        # 2. 利用布尔索引提取。
        # candidate_k_exp[mask_per_batch] 会根据 [B, C] 的 True 位置提取对应的 [1, H, D]
        # 结果形状: [Total_Selected_In_Batch, 1, H, D]
        final_k_raw = candidate_k_exp[mask_per_batch]
        final_v_raw = candidate_v_exp[mask_per_batch]

        # 3. 统计数量（这部分没问题）
        num_selected_chunks_per_sample = mask_per_batch.sum(dim=1)

        # 4. 适配输出格式 [H, Total_Selected_C, D]
        # 先 squeeze(1) 去掉那个大小为 1 的维度 -> [Total_Selected, H, D]
        # 再 permute(1, 0, 2) 交换 Total_Selected 和 H -> [H, Total_Selected, D]
        final_k_to_scatter = final_k_raw.squeeze(1).permute(1, 0, 2).contiguous()
        final_v_to_scatter = final_v_raw.squeeze(1).permute(1, 0, 2).contiguous()

        return final_k_to_scatter, final_v_to_scatter, final_scores, num_selected_chunks_per_sample, winner_global_ids

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        # 转换为tensor
        input_ids_tensor = torch.LongTensor(req.input_ids).to(self.device)
        attention_mask_tensor = torch.LongTensor(req.attention_mask).to(self.device)
        doc_ids_tensor = torch.LongTensor(req.doc_ids).to(self.device)
        position_ids_tensor = torch.LongTensor(req.positions).to(self.device)

        past_key_values: CustomDynamicCache = create_cache()
        past_key_values.meta["require_recall_topk"] = req.require_recall_topk
        past_key_values.meta["qa_mode"] = self.generate_config.qa_mode
        past_key_values.meta["max_generate_tokens"] = self.generate_config.max_generate_tokens
        if self.generate_config.qa_mode:
            past_key_values.meta["tokenizer"] = self.tokenizer
            past_key_values.meta["idx_to_doc"] = self.idx_to_doc
            past_key_values.meta["pattern"] = r"\[(\d+)\]"
            past_key_values.meta["response_string"] = ['' for _ in range(len(req.input_ids))]
            

        # 准备generate kwargs
        for layer_idx in range(self.msa_model_config.num_hidden_layers):
            past_key_values.record_kwargs(layer_idx, {"stage": "prefill_stage2"})

        generate_kwargs = self.generate_kwarg.copy()
        generate_kwargs["past_key_values"] = past_key_values

        inputs = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "doc_ids": doc_ids_tensor,
            "use_cache": True,
            "position_ids": position_ids_tensor,
        }

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
        generated_seqs = self.tokenizer.batch_decode(generated_ids, skip_special_token=True)
        if req.require_recall_topk:
            recall_topks = {layer: args['recall_topk'] for layer, args in past_key_values.cache_kwargs.items() if 'recall_topk' in args}
        else:
            recall_topks = None

        return GenerateResponse(msg_id=req.msg_id,
                                seq_id=req.seq_id,
                                gpu_id=self.gpu_id,
                                generated_texts=generated_seqs, 
                                recall_topk=recall_topks)



class MSAEngine:
    def __init__(self,
                 generate_config: GenerateConfig,
                 model_config: ModelConfig,
                 memory_config: MemoryConfig
                 ):
                 
        # NCCL require global cuda devices, so wen change the device setting here
        # user should set CUDA_VISIBLE_DEVICES envionment vars if they wish to select specific devices
        generate_config.devices = list(range(torch.cuda.device_count()))

        self.device_ids = generate_config.devices
        self.world_size = generate_config.world
        self.generate_config = generate_config
        self.model_config = model_config
        self.memory_config = memory_config

        self.worker_processes = []
        self.worker_qs: Dict[int, mp.Queue] = {}
        self.response_queue = mp.Queue()
        self.request_queue = mp.Queue()
        self.start_workers()

        self.msg_id = 0
        self.nr_active_req = 0
        self.limiter = RequestLimiter(4)
        self.requests: Dict[int, GenerateStub] = {} # msgid -> stub
        self.lock = threading.Lock()
        self.sync_event = threading.Event()
        self.sync_rsp = None
        # Queue for non-generate responses (e.g. AddDocumentsResult) forwarded by receive_response thread
        self.sync_result_queue: queue.Queue = queue.Queue()

        self.initialize()
    
    def initialize(self):
        """初始化Memory服务"""
        print("Initializing Memory service...")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_path)
        self._prepare_template()

        # 加载memory文件
        self._load_memory_file()

        # 先等待所有的worker都就绪了再发送 blocks
        self._wait_ready_signal()
        self._process_buckets()

        # 必须要在bucket处理完成后才启动线程
        self.running = True
        self.recv_rsp_thread = threading.Thread(target=self.receive_response, args=(), daemon=True)
        self.recv_rsp_thread.start()

        self.recv_req_thread = threading.Thread(target=self.receive_request, args=(), daemon=True)
        self.recv_req_thread.start()

        print("MSAEngine initialized")

        
    def _worker_all_gather(self, cmdname: str, timeout=600):
        collected_count = 0
        responses = []

        while collected_count < len(self.worker_qs):
            try:
                response = self.response_queue.get(timeout=timeout)
                assert response.name == cmdname, f"worker all gather got response {response.name} expect {cmdname}"
                responses.append(response)
                collected_count += 1

            except queue.Empty:
                continue

        responses.sort(key=lambda x: x.gpu_id)
        return responses

    def _wait_ready_signal(self):
        collected_count = 0

        while collected_count < len(self.worker_qs):
            ProtocolConstants.expect(self.response_queue, MEMORY_WORKER_READY)
            collected_count += 1

    @staticmethod
    def balanced_bucket_partition(docs: List[Document], n_buckets: int) -> List[List[Document]]:
        """
        全局均衡划分算法：先全局分blocks，再分配到buckets,
        这里并不需要每个 block 必须是最多max_chunk_per_block个 chunk，
        而是根据max_chunk_per_block这个数大致决定分成多少 block（GPU 个数的倍数），然后将这些
        block 做得尽可能 chunk 均衡，最后分配给各个 GPU

        Returns:
                List[List[Document]]: 每个 bucket 分配到的文档列表
        """

        # 全局 blocks 数量
        bucket_docs: List[List[Document]] = [[] for _ in range(n_buckets)] # 每个 bucket 所含的 doc 列表
        bucket_chunk_count = [0] * n_buckets

        def next_bucket():
            return bucket_chunk_count.index(min(bucket_chunk_count))

        for doc in docs:
            bucket_idx = next_bucket()
            bucket_docs[bucket_idx].append(doc)
            bucket_chunk_count[bucket_idx] += doc.num_chunks


        return bucket_docs

    def _sort_reference(self, docs: List[str]) -> Tuple[List[str], List[int], List[int], List[List[int]]]:
        """
        对文档进行重排序并且分 bucket和block，分配到一个 GPU 上的文档被称为 bucket，
        bucket 和 bucket 之间的 chunk 数量尽可能均衡
        Args:
            docs: 文档数据列表

        Returns:
                List[List[Document]]: 每个 bucket 分配到的文档列表
        """
        

        documents: List[Document] = []
        kernel_sz = self.model_config.pooling_kernel_size
        for idx, doc in enumerate(docs):  # idx 是 block 内部的 doc 索引
            new_doc, doc_inputs = compose_input(doc, idx, self.tokenizer)
            length = len(doc_inputs["input_ids"])
            num_chunks = (length + kernel_sz - 1) // kernel_sz
            # print(f"doc {idx} str {len(doc)} id length: {length}, num_chunks: {num_chunks}")

            documents.append(Document(doc=doc, doc_id=idx, num_chunks=num_chunks))
        
        return MSAEngine.balanced_bucket_partition(documents, self.generate_config.world)

    def _load_memory_file(self):
        """加载memory文件"""
        print(f"Loading memory file: {self.memory_config.memory_file_path}")

        if self.memory_config.memory_file_path.endswith('.json'):
            with open(self.memory_config.memory_file_path, 'r') as f:
                data = json.load(f)
            # 处理JSON格式数据
            # 这里需要根据实际的JSON格式来实现
            context_list = data

        elif self.memory_config.memory_file_path.endswith('.pkl'):
            with open(self.memory_config.memory_file_path, 'rb') as f:
                reference_metas = pickle.load(f)

            context_list = list(reference_metas)

        else:
            raise ValueError(f"Unsupported file format: {self.memory_config.memory_file_path}")
        
        # debug function:
        # when env DEBUG_SCALE_MEMORY is set by MemoryFilePath:ScaleFactor
        # we will read from MemoryFilePath and scale it to (ScaleFactor+1) times of memory documents
        # if it is set by ScaleFactor, without MemoryFilePath, we will use self.memory_config.memory_file_path instead
        scale_path = os.environ.get("DEBUG_SCALE_MEMORY", "")
        if scale_path:
            g = scale_path.split(":")
            scale = 0
            if len(g) == 2:
                scale_path = g[0]
                scale=int(g[1])
            elif len(g) == 1:
                scale_path = self.memory_config.memory_file_path
                scale=int(g[0])
            else:
                print("invalid scaling, ignore: ", scale_path)
                
            if scale > 0:
                print(f">>>>>>>>>>>>>>>>>>>> DEBUG: scale from query file {scale_path} with scale {scale}", )
                from src.utils.scale import scale_memory
                scale_memory(context_list, scale_path, scale=scale)

        self.buckets: List[List[Document]] = self._sort_reference(context_list)
        self.docs: List[Document] = []
        for bucket in self.buckets:
            self.docs.extend(bucket)
        chunks = sum(doc.num_chunks for doc in self.docs)

        print(f"Loaded {len(self.docs)} memory document, total chunks: {chunks}")
        for idx, bucket in enumerate(self.buckets):
            print(f"bucket {idx} contains {len(bucket)} documents, {sum(doc.num_chunks for doc in bucket)} chunks")

    def _process_buckets(self):
        """
        发送 blocks 到各 worker
        """

        idx_to_doc = self.get_idx_to_doc()
        for idx, gpu_id in enumerate(self.generate_config.devices):
            ProtocolConstants.send(self.worker_qs[gpu_id],
                                   MEMORY_WORKER_BLOCKS,
                                   data=self.buckets[idx],
                                   block=False)
            ProtocolConstants.send(self.worker_qs[gpu_id],
                                   MEMORY_WORKER_IDX_TO_DOC,
                                   data=idx_to_doc,
                                   block=False)

        # 拿到所有的 doc id bias然后累积更新各 worker
        rsps: List[ReportDocID] = self._worker_all_gather( "report_doc_id", timeout=None)       

        # pooled_doc_id_bias = 0
        # for rsp in rsps:
        #     self.worker_qs[rsp.gpu_id].put(SetDocIDCmd(pooled_doc_id_bias=pooled_doc_id_bias))
        #     pooled_doc_id_bias += rsp.pooled_doc_id_bias
        # _ = self._worker_all_gather("set_doc_id")

    def receive_request(self):
        while self.running:
            try:
                gpu_id, res = self.request_queue.get(timeout=3)
                print(f"Received response from GPU {gpu_id}")
            except queue.Empty:
                continue

    def get_idx_to_doc(self):
        return {doc.doc_id: doc.doc for doc in self.docs}

    @staticmethod
    def service_main(gpu_id: int,
                     request_queue: mp.Queue,
                     response_queue: mp.Queue,
                     generate_config: GenerateConfig,
                     model_config: ModelConfig,
                     memory_config: MemoryConfig
                     ):
        # 1. 初始化 Service (包含 NCCL Init)
        service = MSAService(gpu_id, generate_config, model_config, memory_config)

        print(f"MSAService-{gpu_id} is ready")

        # notify parent I'm ready
        ProtocolConstants.send(response_queue, MEMORY_WORKER_READY, block=False)

        # wait parent to give me blocks to process
        docs: List[Document] = ProtocolConstants.expect(request_queue, MEMORY_WORKER_BLOCKS)
        idx_to_doc: Dict[int, str] = ProtocolConstants.expect(request_queue, MEMORY_WORKER_IDX_TO_DOC)
        service.save_idx_to_doc(idx_to_doc)

        service.generate_blocks(docs)
        del docs
        torch.cuda.empty_cache()

        service.load_model()

        # tell Memory my doc id bias
        response_queue.put(ReportDocID(gpu_id=gpu_id, pooled_doc_id_bias=service.get_max_local_pool_doc_id()), block=False)

        
        while True:
            try:

                cmd: CmdBase = request_queue.get()
                # 检查是否是终止信号
                if cmd is None:
                    if dist.is_initialized():
                        dist.destroy_process_group()
                    return

                elif cmd.name == "generate_request":
                    rsp = service.generate(cmd)
                    response_queue.put(rsp)
                    # torch.cuda.empty_cache() # make perf worse

                elif cmd.name == "add_documents":
                    cmd: AddDocumentsCmd = cmd
                    # Update idx_to_doc in place with new entries
                    if cmd.idx_to_doc_update:
                        service.idx_to_doc.update(cmd.idx_to_doc_update)
                    new_doc_ids = service.add_documents_incremental(cmd.docs)
                    torch.cuda.empty_cache()
                    response_queue.put(AddDocumentsResult(gpu_id=gpu_id, doc_ids=new_doc_ids), block=False)

                elif cmd.name == "reset_memory":
                    # P1.1 rebuild optimization: reset memory bank without checkpoint reload.
                    cmd: ResetMemoryCmd = cmd
                    try:
                        service.reset_memory()
                        if cmd.docs:
                            if cmd.idx_to_doc is not None:
                                service.save_idx_to_doc(cmd.idx_to_doc)
                            service.generate_blocks(cmd.docs)
                            torch.cuda.empty_cache()
                        response_queue.put(
                            ResetMemoryResult(gpu_id=gpu_id, ok=True),
                            block=False,
                        )
                    except Exception as ex:
                        response_queue.put(
                            ResetMemoryResult(gpu_id=gpu_id, ok=False, error=str(ex)),
                            block=False,
                        )

                else:
                    print(f"GPU_ID {gpu_id} recv unknown cmd {cmd.name}")

            except Exception as e:
                print(f"[subprocess {gpu_id}] error: {e}")
                import traceback
                traceback.print_exc()


    def add_documents(self, texts: List[str]) -> List[int]:
        """
        增量添加新文档到已运行的引擎，无需重启。

        流程：
        1. Tokenize 新文档，计算 chunk 数量，分配全局 doc_id
        2. 均衡分配到各 worker（GPU）
        3. 向各 worker 发送 AddDocumentsCmd，等待完成
        4. 更新 MSAEngine 侧的 docs、buckets、idx_to_doc

        Args:
            texts: 新文档的文本列表

        Returns:
            新文档的全局 doc_id 列表（与 texts 顺序对应）
        """
        if not texts:
            return []

        # --- 1. Tokenize 并分配全局 doc_id ---
        kernel_sz = self.model_config.pooling_kernel_size
        # 新文档的 doc_id 从现有文档数量开始
        start_doc_id = len(self.docs)
        new_documents: List[Document] = []
        for i, text in enumerate(texts):
            doc_id = start_doc_id + i
            _, doc_inputs = compose_input(text, doc_id, self.tokenizer)
            length = len(doc_inputs["input_ids"])
            num_chunks = (length + kernel_sz - 1) // kernel_sz
            new_documents.append(Document(doc=text, doc_id=doc_id, num_chunks=num_chunks))

        # --- 2. 均衡分配到各 GPU worker ---
        new_buckets: List[List[Document]] = MSAEngine.balanced_bucket_partition(
            new_documents, self.generate_config.world
        )

        # --- 3. 构造 idx_to_doc 增量更新 ---
        idx_to_doc_update = {doc.doc_id: doc.doc for doc in new_documents}

        # --- 4. 向各 worker 发送 AddDocumentsCmd ---
        for idx, gpu_id in enumerate(self.generate_config.devices):
            cmd = AddDocumentsCmd(docs=new_buckets[idx], idx_to_doc_update=idx_to_doc_update)
            self.worker_qs[gpu_id].put(cmd, block=False)

        # --- 5. 等待所有 worker 完成 ---
        # AddDocumentsResult 由 receive_response 线程转发到 sync_result_queue
        all_new_doc_ids: List[int] = []
        collected = 0
        while collected < len(self.worker_qs):
            rsp: AddDocumentsResult = self.sync_result_queue.get()
            assert isinstance(rsp, AddDocumentsResult), f"unexpected response type: {type(rsp)}"
            all_new_doc_ids.extend(rsp.doc_ids)
            collected += 1

        # --- 6. 更新 MSAEngine 侧状态 ---
        for idx, bucket in enumerate(new_buckets):
            self.buckets[idx].extend(bucket)
        self.docs.extend(new_documents)
        # idx_to_doc on MSAEngine is rebuilt on demand via get_idx_to_doc(),
        # so no separate field to update here.

        # 返回与 texts 顺序一致的 doc_id 列表
        return [doc.doc_id for doc in new_documents]

    def reset_documents(self, new_texts: List[str]) -> None:
        """Replace the entire memory bank with new_texts. Reuses the loaded
        model on each worker (no checkpoint reload), so wall time is bounded
        by the prefill cost on the new doc set rather than the ~70s engine
        re-spawn path.

        流程：
        1. Tokenize new_texts，重新分桶（mirrors _sort_reference）
        2. 向各 worker 发送 ResetMemoryCmd 携带各自 bucket + 全局 idx_to_doc
        3. 等待 ack，更新 MSAEngine 侧 self.docs / self.buckets

        Args:
            new_texts: 完整的新文档文本列表（替换当前 memory bank）

        Raises:
            RuntimeError: 任一 worker 报告 reset 失败
        """
        # --- 1. Tokenize 并重新分桶 ---
        kernel_sz = self.model_config.pooling_kernel_size
        new_documents: List[Document] = []
        for idx, text in enumerate(new_texts):
            _, doc_inputs = compose_input(text, idx, self.tokenizer)
            length = len(doc_inputs["input_ids"])
            num_chunks = (length + kernel_sz - 1) // kernel_sz
            new_documents.append(Document(doc=text, doc_id=idx, num_chunks=num_chunks))

        new_buckets: List[List[Document]] = MSAEngine.balanced_bucket_partition(
            new_documents, self.generate_config.world
        )
        idx_to_doc_full = {doc.doc_id: doc.doc for doc in new_documents}

        # --- 2. 向各 worker 发送 ResetMemoryCmd ---
        for idx, gpu_id in enumerate(self.generate_config.devices):
            cmd = ResetMemoryCmd(docs=new_buckets[idx], idx_to_doc=idx_to_doc_full)
            self.worker_qs[gpu_id].put(cmd, block=False)

        # --- 3. 等待所有 worker 完成 ---
        collected = 0
        errors: List[str] = []
        while collected < len(self.worker_qs):
            rsp: ResetMemoryResult = self.sync_result_queue.get()
            assert isinstance(rsp, ResetMemoryResult), f"unexpected response type: {type(rsp)}"
            if not rsp.ok:
                errors.append(f"gpu {rsp.gpu_id}: {rsp.error}")
            collected += 1

        if errors:
            raise RuntimeError(f"reset_documents failed: {'; '.join(errors)}")

        # --- 4. 更新 MSAEngine 侧状态 ---
        self.buckets = new_buckets
        self.docs = list(new_documents)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_workers()
        return False  # 不抑制异常


    def start_workers(self):
        # 设置启动方式，CUDA 必须用 spawn
        mp.set_start_method('spawn', force=True)

        for gpu_id in self.generate_config.devices:
            request_queue = mp.Queue()
            process = mp.Process(
                target=MSAEngine.service_main,
                args=(gpu_id, request_queue, self.response_queue, self.generate_config, self.model_config, self.memory_config),
                name=f"MSAService-{gpu_id}"
            )
            process.start()
            self.worker_processes.append(process)
            self.worker_qs[gpu_id] = request_queue

    def stop_workers(self):
        print("MSAEngine stop all services")
        self.running = False

        for q in self.worker_qs.values():
            q.put(None)
        
        self.worker_qs = {}

        # 等待所有worker退出
        for process in self.worker_processes:
            process.join()
        self.worker_processes = []

        self.recv_rsp_thread.join()
        self.recv_rsp_thread = None

        self.recv_req_thread.join()
        self.recv_req_thread = None

    def _validate_inputs(self, input_ids: List[List[int]]):
        if any(len(sub) == 0 for sub in input_ids):
            raise ValueError(f"empty query is not acceptable")

        if self.generate_config.max_seq_len > 0:
            seqlen = sum(sum(sub) for sub in input_ids)
            if seqlen > self.generate_config.max_seq_len:
                raise ValueError(f"total sequence length {seqlen} exceeds limitation {self.generate_config.max_seq_len}")

        if self.generate_config.max_query_seq_len > 0:
            max_seq_len = max(sum(sub) for sub in input_ids)
            if max_seq_len > self.generate_config.max_query_seq_len:
                raise ValueError(f"max query sequence length {max_seq_len} exceeds limitation {self.generate_config.max_query_seq_len}")

    def _prepare_template(self):
        response_head = "<|im_start|>"
        response_head_inputs = self.tokenizer(response_head, add_special_tokens=False)
        response_head_input_ids = response_head_inputs["input_ids"]
        response_head_attention_mask = response_head_inputs["attention_mask"]

        pad_token = self.tokenizer.pad_token
        pad_token_id = self.tokenizer.pad_token_id
        prompt_template = self.generate_config.template["prompt"].replace("{prompt}", pad_token)
        prompt_template_inputs = self.tokenizer(prompt_template, add_special_tokens=False)
        prompt_template_input_ids = prompt_template_inputs["input_ids"]
        prompt_template_attention_mask = prompt_template_inputs["attention_mask"]
        pad_index = prompt_template_input_ids.index(pad_token_id)
        template_tail_input_ids = prompt_template_input_ids[pad_index+1:]
        template_tail_attention_mask = prompt_template_attention_mask[pad_index+1:]

        self.prompt_tail_input_ids =  template_tail_input_ids + response_head_input_ids
        self.prompt_tail_attention_mask =  template_tail_attention_mask + response_head_attention_mask
        self.tail_doc_ids = [-2] * (len(template_tail_input_ids)) + [-1] * len(response_head_input_ids)
        
    def _apply_template(self, prompt):
        # question = "\n请根据以上历史文档信息，回答问题\n\n" + prompt + "\n" + f"请返回与问题有关的所有文档\n"
        # question = "\n请根据以上历史文档信息，回答问题\n\n" + prompt + "\n" + f"请返回与问题有关的1个文档\n"
        question = "\nPlease answer the question based on the above historical document information\n\n" + prompt + "\n" + f"Please return all documents related to the question\n"
        prompt_inputs = self.tokenizer(question, add_special_tokens=False)
        prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        input_ids = prompt_inputs["input_ids"] + self.prompt_tail_input_ids
        attention_mask = prompt_inputs["attention_mask"] + self.prompt_tail_attention_mask
        doc_ids = [0] * (len(prompt_inputs["input_ids"])) + self.tail_doc_ids
        return input_ids, attention_mask, doc_ids
    
    def _apply_template_regenerate(self, prompt):
        question = prompt.split("<regenerate>")[1]

        prompt_inputs = self.tokenizer(question, add_special_tokens=False)
        prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        input_ids = prompt_inputs["input_ids"]
        attention_mask = prompt_inputs["attention_mask"]

        # Build doc_ids: 0 for specific regions, -1 for the rest
        doc_ids = [-1] * len(prompt_inputs["input_ids"])

        # Region 1: tokens between the last "]<|object_ref_end|>[" and the first "<|object_ref_end|>" after it
        last_marker_start = question.rfind("]<|object_ref_end|>[")
        if last_marker_start != -1:
            search_start = last_marker_start + len("]<|object_ref_end|>[") - 1
            next_marker = question.find("<|object_ref_end|>", search_start)
            if next_marker != -1:
                region1_text = question[search_start:next_marker]
                region1_tokens = self.tokenizer(region1_text, add_special_tokens=False)["input_ids"]
                prefix_before_region1 = question[:search_start]
                prefix_tokens_len = len(self.tokenizer(prefix_before_region1, add_special_tokens=False)["input_ids"])
                for i in range(prefix_tokens_len, prefix_tokens_len + len(region1_tokens)):
                    if i < len(doc_ids):
                        doc_ids[i] = 0

        # Region 2: "Please return all documents related to the question" and everything before it
        anchor = "Please return all documents related to the question"
        anchor_pos = question.find(anchor)
        if anchor_pos != -1:
            prefix_with_anchor = question[:anchor_pos + len(anchor)]
            prefix_token_len = len(self.tokenizer(prefix_with_anchor, add_special_tokens=False)["input_ids"])
            for i in range(prefix_token_len):
                if i < len(doc_ids):
                    doc_ids[i] = 0

        return input_ids, attention_mask, doc_ids

    def receive_response(self):
        while self.running:
            try:
                rsp = self.response_queue.get(timeout=3)

                # Non-generate responses (e.g. AddDocumentsResult) are forwarded
                # to the synchronous result queue so add_documents() can collect them.
                if not isinstance(rsp, GenerateResponse):
                    self.sync_result_queue.put(rsp)
                    continue

                final_stub = None
                with self.limiter.lock:
                    assert rsp.msg_id in self.requests
                    stub = self.requests[rsp.msg_id]
                    stub.responses.append(rsp)
                    # print(f"Request {rsp.msg_id} got {len(stub.responses)} responses")
                    if len(stub.responses) == self.world_size:
                        final_stub = self.requests.pop(rsp.msg_id)

                if final_stub:
                    # print("Release")
                    self.limiter.release()
                    final_stub.respond()

            except queue.Empty:
                continue

    def default_callback(self, texts: List[str], recall_topk, userdata):
        self.sync_rsp = (texts, recall_topk, userdata)
        self.sync_event.set()

    def generate(self,
                 prompts: Union[str, List[str]],
                 userdata=None,
                 require_recall_topk=False,
                 callback: GenerateReceiver=None):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        world = self.generate_config.world
        bsz = (len(prompts) + world - 1) // world
        if self.generate_config.max_batch_size > 0 and bsz > self.generate_config.max_batch_size:
            raise ValueError(f"Batch size exceeds maximum allowed batch size {self.generate_config.max_batch_size}")
        # Important:
        #     如果输入的 prompts 不够 world size，则需要 pad，我们的方法是用最后一个prompt 填充
        pads = bsz * world - len(prompts)
        if pads > 0:
            prompts += [prompts[-1]] * pads

        batch_input_ids = []
        batch_attention_mask = []
        batch_doc_ids = []
        batch_positions = []

        # template + pooled docs
        current_position = 3 + self.model_config.doc_top_k

        for prompt in prompts:
            if "<regenerate>" in prompt:
                input_ids, attention_mask, doc_ids = self._apply_template_regenerate(prompt)
            else:
                input_ids, attention_mask, doc_ids = self._apply_template(prompt)
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_doc_ids.append(doc_ids)

            # 计算position ids
            position_ids = [current_position + i for i in range(len(input_ids))]
            batch_positions.append(position_ids)
            # current_position += len(input_ids)   # 为什么要累计长度？
        
        # Pad sequences to the same length
        padded_input_ids = []
        padded_attention_mask = []
        padded_doc_ids = []
        padded_position_ids = []

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
        # 使用 doc_end_id 作为 doc_ids 的 padding 值
        # doc_end_id = self.tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"][0]

        for i in range(len(batch_input_ids)):
            if i % bsz == 0:
                max_length = max(len(ids) for ids in batch_input_ids[i:i+bsz])
            input_ids = batch_input_ids[i]
            attention_mask = batch_attention_mask[i]
            doc_ids_seq = batch_doc_ids[i]
            position_ids = batch_positions[i]

            # Pad to max_length
            pad_length = max_length - len(input_ids)
            # 改成left pad，方便position_ids迭代，可直接与自定义的generate兼容
            if pad_length > 0:
                input_ids = [pad_token_id] * pad_length + input_ids
                attention_mask = [0] * pad_length + attention_mask
                doc_ids_seq = [0] * pad_length + doc_ids_seq
                position_ids = [position_ids[0]] * pad_length + position_ids

            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
            padded_doc_ids.append(doc_ids_seq)
            padded_position_ids.append(position_ids)

        # 只检查每个 batch 的第一个 query 就好
        self._validate_inputs(padded_input_ids[0:-1:bsz])


        # TODO: when callback is None
        self.limiter.acquire()

        sync = callback is None
        if callback is None:
            callback = self.default_callback

        self.msg_id += 1
        with self.limiter.lock:
            self.requests[self.msg_id] = GenerateStub(msg_id=self.msg_id,
                                                      userdata=userdata,
                                                      nr_dummy=pads,
                                                      responses=[],
                                                      callback=callback)

        for idx, gpu_id in enumerate(self.generate_config.devices):
            start, end = idx * bsz, (idx+1) * bsz
            req = GenerateRequest(msg_id=self.msg_id,
                                  seq_id=idx,
                                  prompts=prompts[start:end],
                                  input_ids=padded_input_ids[start:end],
                                  attention_mask=padded_attention_mask[start:end],
                                  doc_ids=padded_doc_ids[start:end],
                                  positions=padded_position_ids[start:end],
                                  require_recall_topk=require_recall_topk)
            # print(f"send GPU{gpu_id} batch {len(req.input_ids)}")
            self.worker_qs[gpu_id].put(req, block=False)

        if sync:
            self.sync_event.wait()
            self.sync_event.clear()
            return self.sync_rsp
            
        


    
