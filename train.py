import math, os, time, json, random, sys, datetime, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import torch.distributed as dist
import argparse
from copy import deepcopy
from tqdm import tqdm
from model.transformer import MDMTransformer, MDMConfig
from data import setup_data_bundle
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from typing import Optional, List, Tuple, Union
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf, DictConfig, ListConfig
from model.ema import ExponentialMovingAverage, save_ema_snapshot, save_model_snapshot
from progressive import PhasedMasking, mdm_loss_fn
from eval.sudoku_eval import evaluate_ddp_sudoku
from eval.gsm8k_eval import evaluate_ddp_gsm8k

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str)
    parser.add_argument("--seed", type=int, default=None, help="Optional fixed seed (overrides random)")
    return parser.parse_args()


def setup_ddp():
    if torch.cuda.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank

def evaluate_ddp_dict(model, cfg, device, rank, world_size):
    sampling = cfg.validation.sampling
    if cfg.training.strategy == "arm":
        return {"arm": evaluate_ddp(model, cfg, device, rank, world_size, sampling)}
    base_sampling = sampling
    out = {}

    for confidence in list(base_sampling.confidence):
        for unmasking_num in list(base_sampling.unmasking_num):
            sampling = deepcopy(base_sampling)
            sampling.confidence = confidence
            sampling.unmasking_num = unmasking_num
            out[f"{confidence}_unmasking_{unmasking_num}"] = evaluate_ddp(model, cfg, device, rank, world_size, sampling)
    return out

def grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.norm(p=2).item()
    return total ** 0.5

def evaluate_ddp(model, cfg, device, rank: int, world_size: int, sampling):
    if cfg.data.dataset == "sudoku":
        return evaluate_ddp_sudoku(model, cfg, device, rank, world_size, sampling)
    elif cfg.data.dataset == "tinygsm":
        return evaluate_ddp_gsm8k(model, cfg, device, rank, world_size, sampling)
    else:
        raise ValueError(f"Invalid dataset: {cfg.data.dataset}")

# mdm loss implementation
def mdm_loss(model, input_ids, mask_id: int, prompt_mask: Optional[torch.Tensor] = None, arm_init: bool = False, papl_alpha: Optional[float] = None, papl_tau: float = 1.0):
    # sample integer uniformly for each batch from [1,L]
    # prompt_mask (boolean mask): 1 for prompt
    if prompt_mask is None:
        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    device = input_ids.device
    B, L = input_ids.shape
    L_eff = L - prompt_mask.sum(dim=1 , keepdim=True)
    # uniformly sample the number of positions to mask
    num_mask = torch.floor(torch.rand(B, 1, device=device) * L_eff.clamp(min=1)).long() + 1

    # mask correspondent number of tokens for each batch, 0.0 for the prompt indices
    scores = torch.rand((B, L), device=device).masked_fill(prompt_mask, float('inf')).argsort(dim=1)
    order = scores.argsort(dim=1)
    mask_indices = (order < num_mask)
    masked_input = torch.where(mask_indices, mask_id, input_ids)
    logits = model(masked_input)

    # calculate (reweighted) loss
    num_mask = num_mask.float().expand_as(mask_indices)

    if papl_alpha is not None:
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        target_log_probs = (-nll).detach()
        detached_scores = (target_log_probs / papl_tau).masked_fill(~mask_indices, float("-inf"))
        planner_weights = F.softmax(detached_scores, dim=-1)
        planner_weights = torch.where(mask_indices, planner_weights, torch.zeros_like(planner_weights))
        papl_token_weights = 1.0 + papl_alpha * planner_weights
        loss = papl_token_weights[mask_indices] * nll[mask_indices] / num_mask[mask_indices]
    elif arm_init:
        ce = F.cross_entropy(logits[:, :-1, :][mask_indices[:, 1:]], input_ids[:, 1:][mask_indices[:, 1:]], reduction="none")
        loss = ce / num_mask[mask_indices]
    else:
        ce = F.cross_entropy(logits[mask_indices], input_ids[mask_indices], reduction="none")
        loss = ce / num_mask[mask_indices]
    return loss.sum() / B

def arm_loss(
    model,
    input_ids: torch.Tensor,                    # (B, L)
    eos_id: int,
    prompt_mask: Optional[torch.Tensor] = None, # True = prompt token
):
    if prompt_mask is None:
        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    logits = model(input_ids)          # (B, L, V)
    targets = input_ids[:, 1:]         # (B, L-1)
    pred_logits = logits[:, :-1, :]    # (B, L-1, V)

    valid = ~prompt_mask[:, 1:]        # (B, L-1)

    if eos_id is not None:
        is_eos = (targets == eos_id)               # (B, L-1)
    else:
        is_eos = torch.zeros_like(targets, dtype=torch.bool)
    any_eos = is_eos.any(dim=1)                # (B,)
    first_eos = is_eos.float().argmax(dim=1)   # (B,) 0-based in targets
    first_eos = torch.where(
        any_eos,
        first_eos,
        torch.full_like(first_eos, targets.shape[1] - 1),
    )

    t = torch.arange(targets.shape[1], device=targets.device).unsqueeze(0)  # (1, L-1)
    valid = valid & (t <= first_eos.unsqueeze(1))

    if valid.sum().item() == 0:
        return pred_logits.sum() * 0.0
    return F.cross_entropy(pred_logits[valid], targets[valid], reduction="mean")

# validation loss helper
def val_loss_ddp(model, val_loader, mask_id: int, device, rank: int, world_size: int, strategy: str, eos_id: int, arm_init: bool = False, papl_alpha: Optional[float] = None, papl_tau: float = 1.0):
    model.eval()
    if world_size > 1 and dist.is_initialized() and not isinstance(val_loader.sampler, DistributedSampler):
        sampler = DistributedSampler(val_loader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        val_loader = DataLoader(
        val_loader.dataset,
        batch_size=val_loader.batch_size or 16,
        sampler=sampler,
        num_workers=getattr(val_loader, "num_workers", 4),
        pin_memory=getattr(val_loader, "pin_memory", False),
        drop_last=False,
        )
    else:
        sampler = None

    local_sum = 0.0
    local_count = 0
    local_sum_papl = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc = "Validating", disable = (rank != 0)):
            x0 = batch["labels"].to(device)
            pm = batch["prompt_mask"].to(device) if "prompt_mask" in batch else torch.zeros_like(x0, dtype=torch.bool)
            B = x0.shape[0]

            if strategy == "arm":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    loss = arm_loss(model, x0, eos_id=eos_id, prompt_mask=pm)
                local_sum += float(loss.item() * B)
            elif strategy in ["progressive", "standard"]:
                # inline masking so both losses share one forward pass
                L = x0.shape[1]
                L_eff = L - pm.sum(dim=1, keepdim=True)
                num_mask = torch.floor(torch.rand(B, 1, device=device) * L_eff.clamp(min=1)).long() + 1
                scores = torch.rand((B, L), device=device).masked_fill(pm, float('inf')).argsort(dim=1)
                mask_indices = scores.argsort(dim=1) < num_mask
                masked_input = torch.where(mask_indices, mask_id, x0)
                num_mask_exp = num_mask.float().expand_as(mask_indices)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    logits = model(masked_input)

                if arm_init:
                    ce = F.cross_entropy(logits[:, :-1, :][mask_indices[:, 1:]], x0[:, 1:][mask_indices[:, 1:]], reduction="none")
                    loss = (ce / num_mask_exp[mask_indices]).sum() / B
                else:
                    ce = F.cross_entropy(logits[mask_indices], x0[mask_indices], reduction="none")
                    loss = (ce / num_mask_exp[mask_indices]).sum() / B
                local_sum += float(loss.item() * B)

                if papl_alpha is not None:
                    log_probs = F.log_softmax(logits, dim=-1)
                    nll = -log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
                    target_log_probs = (-nll).detach()
                    detached_scores = (target_log_probs / papl_tau).masked_fill(~mask_indices, float("-inf"))
                    planner_weights = F.softmax(detached_scores, dim=-1)
                    planner_weights = torch.where(mask_indices, planner_weights, torch.zeros_like(planner_weights))
                    papl_token_weights = 1.0 + papl_alpha * planner_weights
                    papl_loss = (papl_token_weights[mask_indices] * nll[mask_indices] / num_mask_exp[mask_indices]).sum() / B
                    local_sum_papl += float(papl_loss.item() * B)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            local_count += B

    tensor = torch.tensor([local_sum, local_count, local_sum_papl], dtype=torch.float, device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_sum, global_count, global_sum_papl = tensor.tolist()

    val_loss = global_sum / max(int(global_count), 1)
    val_loss_reweigh = (global_sum_papl / max(int(global_count), 1)) if papl_alpha is not None else None
    return val_loss, val_loss_reweigh

def parse_k_schedule_increasing(k_schedule) -> List[Tuple[int, int]]:
    """
    Expects k_schedule as an *already increasing* list of [K, step] pairs.
    Validates:
      - each entry is [K, step]
      - steps are strictly increasing
      - (optionally) first step is 0
    Returns list of (K, step) in the same order.
    """
    if k_schedule is None:
        return []

    sched = []
    prev_step = None
    for item in list(k_schedule):
        if not isinstance(item, (list, tuple, ListConfig)) or len(item) != 2:
            raise ValueError(f"k_schedule entries must be [K, step], got {item}")
        K, step = int(item[0]), int(item[1])
        if K <= 0:
            raise ValueError(f"Invalid K in k_schedule: {K}")
        if step < 0:
            raise ValueError(f"Invalid step in k_schedule: {step}")

        if prev_step is not None and step <= prev_step:
            raise ValueError(
                f"k_schedule must have strictly increasing steps, but got step {step} after {prev_step}. "
                f"Full schedule: {list(k_schedule)}"
            )

        sched.append((K, step))
        prev_step = step

    return sched



@torch.no_grad()
def compute_unmask_order(model, x0, mask_id, prompt_mask, unmasking_num=2, confidence="top_k", arm_init=False, micro_batch_size=8):
    """
    Compute unmasking order by progressively revealing GT tokens in order of
    model confidence.  Returns (B, L) int32 numpy array where each value is
    the trajectory step at which that position was unmasked (0 = prompt).
    Processes samples in micro-batches to avoid OOM.
    """
    B = x0.shape[0]
    results = []
    for start in range(0, B, micro_batch_size):
        end = min(start + micro_batch_size, B)
        x0_chunk = x0[start:end]
        pm_chunk = prompt_mask[start:end]
        results.append(_compute_unmask_order_chunk(model, x0_chunk, mask_id, pm_chunk, unmasking_num, confidence, arm_init))
    return np.concatenate(results, axis=0)


@torch.no_grad()
def _compute_unmask_order_chunk(model, x0, mask_id, prompt_mask, unmasking_num=2, confidence="top_k", arm_init=False):
    B, L = x0.shape
    device = x0.device

    xt = x0.clone()
    xt[~prompt_mask] = mask_id

    max_step = (L // max(unmasking_num, 1)) + 2
    unmask_order = torch.zeros((B, L), dtype=torch.int32, device=device)
    unmask_order[~prompt_mask] = max_step

    if arm_init:
        xt_t1 = xt[:, :1].clone()
        xt_work, x0_work = xt[:, 1:].clone(), x0[:, 1:]
        L_work = L - 1
    else:
        xt_work, x0_work = xt, x0
        L_work = L

    step = 0
    for _ in range(L_work // max(unmasking_num, 1) + 1):
        mask_idx = (xt_work == mask_id)
        if mask_idx.sum() == 0:
            break

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            inp = torch.cat([xt_t1, xt_work], dim=1) if arm_init else xt_work
            logits = model(inp)
        if arm_init:
            logits = logits[:, :-1, :]

        p = F.softmax(logits, dim=-1)
        if confidence == "top_k":
            score = torch.where(mask_idx, p.max(dim=-1).values, -float('inf'))
        elif confidence == "top_k_margin":
            top2 = p.topk(k=2, dim=-1).values
            score = torch.where(mask_idx, top2[..., 0] - top2[..., 1], -float('inf'))
        else:
            score = torch.where(mask_idx, p.max(dim=-1).values, -float('inf'))

        step += 1
        for b in range(B):
            k = min(unmasking_num, int(mask_idx[b].sum().item()))
            if k > 0:
                _, sel = torch.topk(score[b], k=k)
                xt_work[b, sel] = x0_work[b, sel]
                offset = 1 if arm_init else 0
                unmask_order[b, sel + offset] = step

    return unmask_order.cpu().numpy()


def main(cfg: DictConfig, args):
    # setup the DDP
    rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)
    if is_main:
        print("Hey, we start training!")
        print(f"Training with {world_size} GPUs")

    base_seed = args.seed if args.seed is not None else random.randint(0, int(1e8))
    seed = base_seed + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    # ckpt dir
    ckpt_dir = f"/n/netscratch/dam_lab/Lab/jgeuter/MDMPre/ckpts/date={datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    os.makedirs(ckpt_dir, exist_ok=True)
    if is_main:
        print(f"Checkpoints will be saved to: {ckpt_dir}")

    # set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Initialize the model
    model_cfg_dict = cfg.model
    model_config = MDMConfig(**model_cfg_dict)
    model = MDMTransformer(model_config).to(device)

    # ARM initialization
    arm_init_path = model_cfg_dict.get("arm_init", "none")
    if arm_init_path != "none":
        model_config.predict_next_token = True
        if is_main:
            print(f"Initializing MDM from ARM checkpoint: {arm_init_path}")
        arm_ckpt = torch.load(arm_init_path, map_location="cpu")
        sd = arm_ckpt.get("model_state_dict", arm_ckpt)
        model.load_state_dict(sd, strict=True)


    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model is ready, parameters: {num_params/1e6:.2f}M")

    # model wrapping
    if world_size > 1 and torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            print(f"Model wrapping is done!")

    # data
    data_cfg = cfg.data
    train_cfg = cfg.training
    assert train_cfg.save_steps % train_cfg.eval_steps == 0, "save_steps must be divisible by eval_steps"

    papl_reweigh = getattr(train_cfg, 'papl_reweigh', False)
    papl_alpha = float(getattr(train_cfg, 'papl_alpha', 1.0)) if papl_reweigh else None
    papl_tau = float(getattr(train_cfg, 'papl_tau', 1.0))
    val_cfg = cfg.validation
    data_bundle = setup_data_bundle(data_cfg)
    train_loader, val_loader = data_bundle.train_loader, data_bundle.val_loader    
    mask_id = data_cfg.mask_id
    eos_id = getattr(val_cfg.sampling, "eos_id", None)

    # training hyperparemeters
    # attach DDP sampler
    if world_size > 1 and torch.cuda.is_available():
        train_sampler = DistributedSampler(
            train_loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_cfg.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=False,
            drop_last=False
        )
    else:
        train_sampler = None

    # optimizer and scheduler
    max_steps = getattr(train_cfg, 'max_steps', None)
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    num_training_steps = train_cfg.num_epochs * len(train_loader)
    if max_steps is not None:
        num_training_steps = min(num_training_steps, max_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_cfg.warmup_steps, num_training_steps=num_training_steps)
    if train_cfg.ema is not None:
        assert 0.0 < train_cfg.ema < 1.0, "EMA decay must be between 0 and 1"
        model_to_ema = model.module if isinstance(model, DDP) else model
        ema_params = [p for p in model_to_ema.parameters() if p.requires_grad]
        ema = ExponentialMovingAverage(ema_params, decay=train_cfg.ema)
        if is_main:
            print("EMA is enabled with decay:", train_cfg.ema)

    strategy = train_cfg.strategy
    puma_warmup_steps = 0  # default; overridden below for progressive strategy
    warmup_iter = None

    # k schedule for progressive unmasking. If None use fixed K. If "linear", linearly increase the unmasking steps from 1 to K over the training steps.
    # If a list of integers, use the list as the k_steps. If an integer, use constant interval increase.
    if strategy == "progressive":
        k_schedule = parse_k_schedule_increasing(getattr(train_cfg, "k_schedule", None))
        if len(k_schedule) == 0:
            k_schedule = [(train_cfg.K, 0)]
        
        current_k = k_schedule[0][0]

        if is_main:
            print("Using K Schedule:")
            for K, step in k_schedule:
                print(f"Step {step}: K={K}")

        # intialize the pool
        B = train_cfg.batch_size
        L = model_config.max_position
        def make_pool(K):
            B = train_cfg.batch_size
            L = model_config.max_position
            return PhasedMasking(
                train_loader, B, mask_id, K, device, L,
                mode=train_cfg.mode,
                confidence_threshold=train_cfg.confidence_threshold,
                eos_id=train_cfg.eos_id,
                random_unmask_prob=getattr(train_cfg, 'random_unmask_prob', 0.0),
                prompt_only_reset=getattr(cfg.data, 'single_seq_sudoku', False),
            )

        # warmup: use vanilla MDM training before switching to progressive
        puma_warmup_ratio = getattr(train_cfg, 'puma_warmup_ratio', 0.0)
        puma_warmup_steps = int(puma_warmup_ratio * num_training_steps)
        warmup_iter = None

        if puma_warmup_steps == 0:
            pool = make_pool(current_k)
        else:
            pool = None  # will be created after warmup
            warmup_iter = iter(train_loader)
            if is_main:
                print(f"PUMA warmup: {puma_warmup_steps} steps of vanilla MDM training before switching to progressive")

        next_k_idx = 1


    # training loop
    global_step = 0

        
    # wandb initialize
    if cfg.wandb.wandb and is_main:
        run_name = cfg.wandb.name
        wandb.init(project=cfg.wandb.project, name=run_name, entity=cfg.wandb.entity)

    # --- trajectory logging setup ---
    traj_log_steps = getattr(train_cfg, 'traj_log_steps', 0)
    traj_log_data = None
    traj_save_path = None
    traj_x0 = None
    traj_pm = None

    if traj_log_steps > 0 and is_main:
        traj_num_samples = getattr(train_cfg, 'traj_num_samples', 100)
        if cfg.data.dataset == "sudoku":
            val_dir = cfg.validation.val_dir
            test_file = os.path.join(val_dir, "test_mdm.npy")
            raw = np.load(test_file)[:traj_num_samples]
            traj_x0 = torch.from_numpy(raw).long().to(device)
            traj_pm = torch.zeros(traj_x0.shape, dtype=torch.bool, device=device)
            traj_pm[:, :81] = True
        else:
            ds = data_bundle.val_loader.dataset
            n_traj = min(traj_num_samples, len(ds))
            samples = [ds[i] for i in range(n_traj)]
            traj_x0 = torch.stack([s["labels"] for s in samples]).to(device)
            traj_pm = torch.stack([s["prompt_mask"] for s in samples]).bool().to(device)

        traj_log_data = {
            "mask_id": int(mask_id),
            "dataset": cfg.data.dataset,
            "prompt_masks": traj_pm.cpu().numpy(),
            "unmask_orders": {},
        }
        traj_save_path = os.path.join(ckpt_dir, "traj_log.npy")
        print(f"Trajectory logging: {traj_x0.shape[0]} samples, every {traj_log_steps} steps")
        print(f"Saving to: {traj_save_path}")

        # log step 0 (initial model)
        model.eval()
        m = model.module if isinstance(model, DDP) else model
        order_0 = compute_unmask_order(m, traj_x0, mask_id, traj_pm,
                                       arm_init=model_config.predict_next_token)
        traj_log_data["unmask_orders"][0] = order_0
        np.save(traj_save_path, traj_log_data)
        print("Trajectory logged at step 0")
        model.train()

    # save step-0 checkpoint (before any training)
    if is_main:
        if train_cfg.ema is not None:
            saved_path = save_ema_snapshot(ckpt_dir, model, ema, cfg, 0, 0, None, None)
            if saved_path is not None:
                print(f"Step-0 EMA snapshot saved to: {saved_path}")
        saved_path = save_model_snapshot(ckpt_dir, model, cfg, 0, 0, val_loss=None)
        if saved_path is not None:
            print(f"Step-0 model saved to: {saved_path}")

    for epoch in range(train_cfg.num_epochs):
        model.train()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if strategy == "progressive":
            if pool is not None:
                pool.reset_loader_iter()
            if warmup_iter is not None and global_step < puma_warmup_steps:
                warmup_iter = iter(train_loader)
            steps_per_epoch = len(train_loader)
            iterable = range(steps_per_epoch)
        elif strategy == "standard" or strategy == "arm":
            iterable = train_loader

        if is_main:
            pbar = tqdm(iterable, desc=f"Epoch {epoch+1}")
        else:
            pbar = iterable

        for itr in pbar:
            # update current K if using k schedule (skip during warmup)
            if strategy == "progressive" and global_step >= puma_warmup_steps and next_k_idx < len(k_schedule) and global_step == k_schedule[next_k_idx][1]:
                current_k = k_schedule[next_k_idx][0]
                if is_main:
                    print(f"[K-SWITCH] Step {global_step}: K={current_k}")

                pool = make_pool(current_k)
                pool.reset_loader_iter()
                next_k_idx += 1

            # to enable flashattention, we do the autocast
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled = torch.cuda.is_available()):
                if strategy == "progressive" and global_step < puma_warmup_steps:
                    # vanilla MDM training during warmup
                    try:
                        batch = next(warmup_iter)
                    except StopIteration:
                        warmup_iter = iter(train_loader)
                        batch = next(warmup_iter)
                    input_ids = batch["labels"].to(device)
                    prompt_mask = batch["prompt_mask"].to(device) if "prompt_mask" in batch else None
                    loss = mdm_loss(model, input_ids, mask_id, prompt_mask=prompt_mask, arm_init=model_config.predict_next_token, papl_alpha=papl_alpha, papl_tau=papl_tau)
                elif strategy == "progressive":
                    if pool is None:
                        pool = make_pool(current_k)
                        if is_main:
                            print(f"[WARMUP->PUMA] Switching to progressive at step {global_step}")
                    xt = pool.current_batch()
                    logits = model(xt)
                    log_probs = F.log_softmax(logits, dim=-1)
                    loss = mdm_loss_fn(log_probs, pool.x0, pool.xt, mask_id, prompt_mask=pool.state['prompt_mask'], arm_init=model_config.predict_next_token, papl_alpha=papl_alpha, papl_tau=papl_tau)
                elif strategy == "standard":
                    batch = itr
                    input_ids = batch["labels"].to(device)
                    prompt_mask = batch["prompt_mask"].to(device) if "prompt_mask" in batch else None
                    loss = mdm_loss(model, input_ids, mask_id, prompt_mask=prompt_mask, arm_init=model_config.predict_next_token, papl_alpha=papl_alpha, papl_tau=papl_tau)
                elif strategy == "arm":
                    batch = itr
                    input_ids = batch["labels"].to(device)
                    prompt_mask = batch["prompt_mask"].to(device) if "prompt_mask" in batch else None
                    loss = arm_loss(model, input_ids, eos_id=eos_id, prompt_mask=prompt_mask)
                else:
                    raise ValueError(f"Invalid training strategy: {strategy}")
            
            optimizer.zero_grad()

            loss.backward()
            if train_cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            optimizer.step()
            if train_cfg.ema is not None:
                ema.update(ema_params)
            scheduler.step()
            global_step += 1
            
            # update a new seq
            if strategy == "progressive" and global_step > puma_warmup_steps and pool is not None:
                pool.update_with_logits(log_probs)

            if is_main:
                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

                if global_step % train_cfg.logging_steps == 0:
                    print(f"Epoch {epoch+1}, Step {global_step}, Loss {loss.item()}")
                    if cfg.wandb.wandb:
                        wandb.log({"loss": loss.item()}, step=global_step)

                        gn = grad_norm(model.parameters())
                        wandb.log({"grad_norm": gn}, step=global_step)

                        if strategy == "progressive":
                            wandb.log({"current_k": current_k}, step=global_step)

            if global_step % train_cfg.eval_steps == 0:
                model.eval()

                # trajectory logging
                if traj_log_data is not None and global_step % traj_log_steps == 0:
                    m = model.module if isinstance(model, DDP) else model
                    order = compute_unmask_order(m, traj_x0, mask_id, traj_pm,
                                                arm_init=model_config.predict_next_token)
                    traj_log_data["unmask_orders"][global_step] = order
                    np.save(traj_save_path, traj_log_data)
                    print(f"Trajectory logged at step {global_step}")

                # validaton on the downstream task; disabled when we use EMA
                if train_cfg.ema is None:
                    val_acc_dict = evaluate_ddp_dict(model, cfg, device, rank, world_size)
                else:
                    val_acc_dict = None

                # validation loss (mdm loss on the validation dataset)
                val_loss, val_loss_reweigh = val_loss_ddp(model, val_loader, mask_id, device, rank, world_size, strategy, eos_id, arm_init=model_config.predict_next_token, papl_alpha=papl_alpha, papl_tau=papl_tau)

                # EMA evaluation
                if train_cfg.ema is not None:
                    torch.cuda.empty_cache()
                    model_to_ema = model.module if isinstance(model, DDP) else model
                    ema.store(model_to_ema.parameters())
                    ema.copy_to(model_to_ema.parameters())
                   
                    with torch.inference_mode():
                        # validaton on the downstream task
                        val_acc_dict = evaluate_ddp_dict(model, cfg, device, rank, world_size)
                    ema.restore(model_to_ema.parameters())
                
                if is_main:
                    # eval acc logging
                    for key, value in val_acc_dict.items():
                        print(f"Epoch {epoch+1}, Step {global_step}, Validation Accuracy {key}: {value}")
                        if cfg.wandb.wandb:
                            if train_cfg.ema is not None:
                                wandb.log({"ema_val_acc_" + key: value}, step=global_step)
                            else:
                                wandb.log({"val_acc_" + key: value}, step=global_step)
                    
                    # validation loss logging
                    print(f"Epoch {epoch+1}, Step {global_step}, Validation Loss: {val_loss}")
                    if cfg.wandb.wandb:
                        wandb.log({"val_loss": val_loss}, step=global_step)
                        if val_loss_reweigh is not None:
                            wandb.log({"val_loss_reweigh": val_loss_reweigh}, step=global_step)

                    if is_main and global_step % train_cfg.save_steps == 0 and train_cfg.ema is not None:
                        saved_path = save_ema_snapshot(ckpt_dir, model, ema, cfg, epoch, global_step, val_loss, val_acc_dict)
                        if saved_path is not None:
                            print(f"EMA Model saved to: {saved_path}")

                    if is_main and global_step % train_cfg.save_steps == 0:
                        # save non-EMA snapshot
                        saved_path = save_model_snapshot(
                            ckpt_dir, model, cfg, epoch, global_step,
                            val_loss=val_loss,
                            extra=val_acc_dict,
                        )
                        if saved_path is not None:
                            print(f"Model saved to: {saved_path}")
                
                model.train()

            if max_steps is not None and global_step >= max_steps:
                if is_main:
                    print(f"Reached max_steps={max_steps}, stopping.")
                break

        # break out of epoch loop too
        if max_steps is not None and global_step >= max_steps:
            break

    # final trajectory log
    if traj_log_data is not None and global_step not in traj_log_data["unmask_orders"]:
        model.eval()
        m = model.module if isinstance(model, DDP) else model
        order = compute_unmask_order(m, traj_x0, mask_id, traj_pm,
                                    arm_init=model_config.predict_next_token)
        traj_log_data["unmask_orders"][global_step] = order
        np.save(traj_save_path, traj_log_data)
        print(f"Final trajectory logged at step {global_step}")

    # final checkpoint if last step wasn't a save point
    if is_main and global_step % train_cfg.save_steps != 0:
        if train_cfg.ema is not None:
            saved_path = save_ema_snapshot(ckpt_dir, model, ema, cfg, epoch, global_step, None, None)
            if saved_path is not None:
                print(f"Final EMA snapshot saved to: {saved_path}")
        saved_path = save_model_snapshot(ckpt_dir, model, cfg, epoch, global_step, val_loss=None)
        if saved_path is not None:
            print(f"Final model saved to: {saved_path}")

    if cfg.wandb.wandb and is_main:
        wandb.finish()
    
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    cfg_path = args.cfg
    cfg = OmegaConf.load(cfg_path)
    main(cfg, args)
