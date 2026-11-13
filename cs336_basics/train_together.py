from typing import IO, BinaryIO
import os
import torch
import numpy as np

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.AdamW import AdamW
from cs336_basics.data_loading import dataLoading
from cs336_basics.cross_entropy import crossEntropy
from cs336_basics.gradient_clipping import gradientClipping
from cs336_basics.lr import lrCosineSchedule
from cs336_basics.checkpoint import saveCheckPoint

def trainTogther(
    train_src: str | os.PathLike,
    valid_src: str | os.PathLike,
    out_dir: str | os.PathLike,
    device: str | None = "cpu",
):
    # 模型超参
    vocab_size = 10000
    context_length = 256
    d_model = 512
    num_layers = 4
    num_heads = 16
    d_ff = 1344
    rope_theta = 10000.0

    # 训练超参
    BATCH_SIZE = 32
    STEPS = 200
    LR_MAX = 1e-3
    LR_MIN = 1e-4
    WARMUP_ITERS = 50
    COSINE_CYCLE_ITERS = STEPS
    MAX_L2_NORM = 1.0
    WEIGHT_DECAY = 0.01
    BETAS = (0.9, 0.999)
    EPS = 1e-8
    EVAL_INTERVAL = 50
    SAVE_INTERVAL = 100
    EVAL_BATCHES = 50

    model = TransformerLM(vocab_size=vocab_size, context_length=context_length, d_model=d_model, num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, rope_theta=rope_theta, device=device)
    opt = AdamW(params=model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY, betas=BETAS, eps=EPS)

    train_data = np.load(train_src, mmap_mode="r")
    valid_data = np.load(valid_src, mmap_mode="r")

    x, y = dataLoading(train_data, BATCH_SIZE, context_length, device)
    logits = model(x)
    loss = crossEntropy(logits.view(-1, vocab_size), y.view(-1))
    valid_x, valid_y = dataLoading(valid_data, EVAL_BATCHES, context_length, device)

    print("------- init -------")
    print("train tokens:", len(train_data))
    print("params:", sum(p.numel() for p in model.parameters()))
    print("logits:", logits.shape, "loss:", loss.item())

    print("------- start -------")
    for it in range(STEPS):
        x, y = dataLoading(train_data, BATCH_SIZE, context_length, device)
        opt.zero_grad()
        logits = model(x)
        loss = crossEntropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        gradientClipping(model.parameters(), MAX_L2_NORM)
        lr = lrCosineSchedule(it, LR_MAX, LR_MIN, WARMUP_ITERS, COSINE_CYCLE_ITERS)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.step()

        if (it + 1) % 10 == 0:
            print("step: ", it + 1, ", loss:", loss.item(), ", lr:", lr)

        with torch.no_grad():
            if (it + 1) % EVAL_INTERVAL == 0:
                eval_logits = model(valid_x)
                eval_loss = crossEntropy(eval_logits.view(-1, vocab_size), valid_y.view(-1))
                print("step: ", it + 1, " eval_loss: ", eval_loss.item())

            if (it + 1) % SAVE_INTERVAL == 0:
                saveCheckPoint(model, opt, it + 1, out_dir + f"ckpt_{it+1:07d}.pt")
