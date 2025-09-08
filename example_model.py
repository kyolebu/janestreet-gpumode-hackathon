#!/usr/bin/env python3
"""
Example model implementation for GPU inference game.
Shows how to implement the BaseInferenceClient.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
import argparse
from typing import Dict, List, Union, Tuple, Any
import torch

from huggingface_hub import hf_hub_download

from client import BaseInferenceClient, PendingRequest, InferenceResponse
from model.inference_model import MultiTowerModel, ModelConfig
def is_same_structure_and_shape(a, b) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        return all(is_same_structure_and_shape(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return all(is_same_structure_and_shape(a[k], b[k]) for k in a.keys())
    return False

def copy_state_recursively(dst, src):
    if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor):
        # Avoids realloc, stays on device, no sync required
        dst.copy_(src, non_blocking=True)
    elif isinstance(dst, (list, tuple)) and isinstance(src, (list, tuple)):
        for d, s in zip(dst, src):
            copy_state_recursively(d, s)
    elif isinstance(dst, dict) and isinstance(src, dict):
        for k in dst.keys():
            copy_state_recursively(dst[k], src[k])
    else:
        # Fallback if structure diverged
        raise ValueError("State structures differ; cannot copy in-place.")

def stack_states_recursively(states_list: List[Any]) -> Any:
    """
    Given a list of state-structures (each with batch dim == 1), recursively
    stack them along the batch dimension to make a single batched state (B, ...).
    """
    if isinstance(states_list[0], torch.Tensor):
        # All tensors must have shape [1, ...]; concat to [B, ...]
        return torch.cat(states_list, dim=0)
    elif isinstance(states_list[0], (list, tuple)):
        # Zip over children and recurse
        zipped = zip(*states_list)
        stacked = [stack_states_recursively(list(items)) for items in zipped]
        return type(states_list[0])(stacked)
    elif isinstance(states_list[0], dict):
        # Recurse key-by-key (keys assumed identical)
        keys = states_list[0].keys()
        return {
            k: stack_states_recursively([st[k] for st in states_list]) for k in keys
        }
    else:
        # Non-tensor leaves (e.g., None); take from the first
        return states_list[0]

def clone_state_recursively(state: Any) -> Any:
    """Deep clone of a state structure; tensors become detached, contiguous clones."""
    if isinstance(state, torch.Tensor):
        return state.detach().contiguous().clone()
    elif isinstance(state, (list, tuple)):
        return type(state)(clone_state_recursively(s) for s in state)
    elif isinstance(state, dict):
        return {k: clone_state_recursively(v) for k, v in state.items()}
    else:
        return state

def index_state_recursively(state_batched: Any, idx: int) -> Any:
    """
    Select a single element (keeping batch dim) from a batched state.
    Returns a view; pair with clone_state_recursively before persisting.
    """
    if isinstance(state_batched, torch.Tensor):
        return state_batched[idx:idx+1, ...]
    elif isinstance(state_batched, (list, tuple)):
        return type(state_batched)(index_state_recursively(s, idx) for s in state_batched)
    elif isinstance(state_batched, dict):
        return {k: index_state_recursively(v, idx) for k, v in state_batched.items()}
    else:
        return state_batched

def expand_state_recursively(state: any, batch_size: int) -> any:
    """Recursively traverses a state object and expands any tensors it finds."""
    if isinstance(state, torch.Tensor):
        # Base case: we found a tensor, so expand it.
        return state.expand(batch_size, *state.shape[1:])
    elif isinstance(state, (list, tuple)):
        # Recursive step: it's a list or tuple, so process each element.
        return type(state)(expand_state_recursively(s, batch_size) for s in state)
    else:
        # It's something else (None, int, etc.), so return it unchanged.
        return state

def reduce_state_recursively(state: any) -> any:
    """Recursively traverses a state object and reduces the batch dim of any tensors it finds to 1."""
    if isinstance(state, torch.Tensor):
        # Base case: we found a tensor, so slice it to keep only the last item.
        # The `[-1:, ...]` slice keeps the batch dimension.
        return state[-1:, ...]
    elif isinstance(state, (list, tuple)):
        # Recursive step: it's a list or tuple, so process each element.
        return type(state)(reduce_state_recursively(s) for s in state)
    else:
        # It's something else (None, int, etc.), so return it unchanged.
        return state

# --- Main Client Implementation ---

def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class NnInferenceClient(BaseInferenceClient):
    def __init__(
        self,
        num_symbols: int,
        server_host: str = "localhost",
        server_port: int = 8080,
        device: str | None = None,
        token: str | None = None,
        batch_size: int = 256,
    ):
        super().__init__(num_symbols, server_host, server_port)
        self.device = device or get_default_device()
        self.batch_size = batch_size

        # 1. Set matmul precision for performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('medium')

        # 2. Define the model configuration
        config = ModelConfig(
            hidden_size=2048,
            proj_size=4096,
            tower_depth=12,
            num_heads=8,
            num_features=79,
        )
        
        # 3. Create the model instance
        self.model = MultiTowerModel(config).to(self.device)

        # 4. Now you can safely calculate the number of parameters
        nparams = sum(p.numel() for p in self.model.parameters())
        print(f"{nparams = }")

        # 5. Load the pre-trained weights
        weights_file = hf_hub_download(
            repo_id="jane-street-gpu-mode/hackathon",
            filename="state_dict.pt",
            token=token,
        )
        weights = torch.load(weights_file, weights_only=True)
        self.model.load_state_dict(weights)

        # 6. Compile the model after loading weights
        self.model = torch.compile(
            self.model,
            mode="reduce-overhead",
            fullgraph=True,
        )

        # 7. Initialize the model states with batch_size=1
        self.states = {
            f"SYM_{num:03d}": self.model.init_state(1, self.device)
            for num in range(self.num_symbols)
        }

    def process_batch(self, requests_by_symbol: Dict[str, List[PendingRequest]]) -> InferenceResponse:
        # -------- Flatten (preserve order) --------
        unique_ids: List[str] = []
        symbols: List[str] = []
        requests: List[PendingRequest] = []
        for sym, reqs in requests_by_symbol.items():
            if not reqs:
                continue
            symbols.extend([sym] * len(reqs))
            requests.extend(reqs)

        n_req = len(requests)
        print(f"Received {n_req} requests")
        if n_req == 0:
            return InferenceResponse(unique_ids=[], predictions=[], client_timestamp=time.time())

        # -------- Accumulate preds on device (avoid per-batch .cpu/.tolist) --------
        preds_device: List[torch.Tensor] = []
        unique_ids_extend = unique_ids.extend  # micro-opt

        # CUDA timing without host syncs
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

        # Hot helpers
        device = self.device
        non_blocking = True

        # If you know feature width F, cache it
        # (otherwise infer from first request in each batch below)
        F_cached = None
        if requests:
            F_cached = len(requests[0].features)

        for i in range(0, n_req, self.batch_size):
            batch_requests = requests[i : i + self.batch_size]
            symbols_in_batch = symbols[i : i + self.batch_size]
            B = len(batch_requests)

            # ---- Build features with pinned staging + async H2D ----
            F = F_cached if F_cached is not None else len(batch_requests[0].features)
            # Pinned CPU buffer
            features_cpu = torch.empty((B, F), dtype=torch.float32, pin_memory=True)
            # Vectorized fill from NumPy is faster than Python loop if you can afford one temporary
            # (If req.features is already a NumPy array, skip the np.stack)
            features_cpu.copy_(torch.from_numpy(np.stack([r.features for r in batch_requests], axis=0)))
            # Async H2D
            features_batch = features_cpu.to(device, non_blocking=non_blocking)

            # ---- Stack states (already on device) ----
            per_pos_states = [self.states[sym] for sym in symbols_in_batch]
            state_for_batch = stack_states_recursively(per_pos_states)  # keep this purely view/concat on device

            with torch.inference_mode():
                torch.compiler.cudagraph_mark_step_begin()
                pred_batch, new_state_from_batch = self.model(features_batch, state_for_batch)

            # Defer host materialization: keep on device
            if isinstance(pred_batch, torch.Tensor):
                # Make sure contiguous to avoid later cat penalties
                preds_device.append(pred_batch.contiguous())
            else:
                # If your model can return list/np, coerce once to device tensor
                preds_device.append(torch.as_tensor(pred_batch, device=device))

            # ---- Reduce updated states back in-place (avoid allocator churn) ----
            # Only keep the last occurrence per symbol
            last_index_for_symbol: Dict[str, int] = {}
            for j, sym in enumerate(symbols_in_batch):
                last_index_for_symbol[sym] = j

            for sym, j in last_index_for_symbol.items():
                updated_view = index_state_recursively(new_state_from_batch, j)  # view on device
                # Prefer in-place copy into existing storage to avoid replacing tensors
                if sym in self.states and is_same_structure_and_shape(self.states[sym], updated_view):
                    copy_state_recursively(self.states[sym], updated_view)
                else:
                    # Fallback if shape/structure changed (try to avoid during CUDA graphs)
                    self.states[sym] = clone_state_recursively(updated_view)

            # IDs (pure CPU book-keeping)
            unique_ids_extend([r.unique_id for r in batch_requests])

        # ---- Single D2H at the end ----
        end_evt.record()

        if len(preds_device) == 1:
            pred_all = preds_device[0]
        else:
            pred_all = torch.cat(preds_device, dim=0)

        # Async D2H into pinned CPU, then just one sync
        pred_cpu = pred_all.to("cpu", non_blocking=True)
        torch.cuda.current_stream().synchronize()

        # Now it’s safe to materialize Python
        preds: List[List[float]] | List[float]
        if pred_cpu.ndim == 1:
            preds = pred_cpu.tolist()
        else:
            preds = pred_cpu.tolist()

        # Non-blocking timing (we already synced once above for the copy)
        elapsed_ms = start_evt.elapsed_time(end_evt)  # ms
        if preds:
            per_item = (elapsed_ms / 1000.0) / len(preds)
            print(f"len(preds) = {len(preds)}, elapsed = {elapsed_ms/1000:.6f}, per_item = {per_item:.6f}")

        return InferenceResponse(unique_ids=unique_ids, predictions=preds, client_timestamp=time.time())


def main():
    parser = argparse.ArgumentParser(description="Example inference client")
    parser.add_argument("--host", type=str, default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument(
        "--num-symbols",
        type=int,
        default=20,
        help="Number of symbols in the tradeable universe",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token to download the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for processing requests",
    )

    args = parser.parse_args()
    client = NnInferenceClient(
        num_symbols=args.num_symbols,
        server_host=args.host,
        server_port=args.port,
        token=args.token,
        batch_size=args.batch_size,
    )

    client.run()


if __name__ == "__main__":
    main()