import argparse
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from operator import attrgetter
from typing import Dict, List, Optional, Union

import safetensors
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel

from peft import LoHaConfig, LoKrConfig, LoraConfig, PeftType, get_peft_model, set_peft_model_state_dict
from peft.tuners.lokr.layer import factorization

UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
PREFIX_UNET = "lora_unet"
PREFIX_TEXT_ENCODER = "lora_te"

@dataclass
class LoRAInfo:
    kohya_key: str
    peft_key: str
    alpha: Optional[float] = None
    rank: Optional[int] = None
    lora_A: Optional[torch.Tensor] = None
    lora_B: Optional[torch.Tensor] = None

    def pef_state_dict(self) -> Dict[str, torch.Tensor]:
        if self.lora_A is None or self.lora_B is None:
            raise ValueError("At least one of lora_A or lora_B is None, they must both be provided")
        return {
            f"base_model.model.{self.peft_key}.lora_A.weight": self.lora_A,
            f"base_model.model.{self.peft_key}.lora_B.weight": self.lora_B
        }

@dataclass
class LoHaInfo:
    kohya_key: str
    peft_key: str
    alpha: Optional[float] = None
    rank: Optional[int] = None
    hada_w1_a: Optional[torch.Tensor] = None
    hada_w1_b: Optional[torch.Tensor] = None
    hada_w2_a: Optional[torch.Tensor] = None
    hada_w2_b: Optional[torch.Tensor] = None
    hada_t1: Optional[torch.Tensor] = None
    hada_t2: Optional[torch.Tensor] = None

    def peft_state_dict(self) -> Dict[str, torch.Tensor]:
        if self.hada_w1_a is None or self.hada_w1_b is None or self.hada_w2_a is None or self.hada_w2_b is None:
            raise ValueError("At least one of hada_w1_a or hada_w1_b or hada_w2_a or hada_w2_b is None, they must all be provided")

        state_dict = {
            f"base_model.model.{self.peft_key}.hada_w1_a": self.hada_w1_a,
            f"base_model.model.{self.peft_key}.hada_w1_b": self.hada_w1_b,
            f"base_model.model.{self.peft_key}.hada_w2_a": self.hada_w2_a,
            f"base_model.model.{self.peft_key}.hada_w2_b": self.hada_w2_b
        }

        if not (
                (self.hada_t1 is None and self.hada_t2 is None) or (self.hada_t1 is not None and self.hada_t2 is not None)
        ):
            raise ValueError("hada_t1 and hada_t2 must both be provided or both be None")

        if self.hada_t1 is not None and self.hada_t2 is not None:
            state_dict[f"base_model.model.{self.peft_key}.hada_t1"] = self.hada_t1
            state_dict[f"base_model.model.{self.peft_key}.hada_t2"] = self.hada_t2

        return state_dict

@dataclass
class LoKrInfo:
    kohya_key: str
    peft_key: str
    alpha: Optional[float] = None
    rank: Optional[int] = None
    lokr_w1: Optional[torch.Tensor] = None
    lokr_w1_a: Optional[torch.Tensor] = None
    lokr_w1_b: Optional[torch.Tensor] = None
    lokr_w2: Optional[torch.Tensor] = None
    lokr_w2_a: Optional[torch.Tensor] = None
    lokr_w2_b: Optional[torch.Tensor] = None
    lokr_t2: Optional[torch.Tensor] = None

    def peft_state_dict(self) -> Dict[str, torch.Tensor]:
        if (self.lokr_w1 is None) and ((self.lokr_w1_a is None) or (self.lokr_w1_b is None)):
            raise ValueError("Either lokr_w1 or both lokr_w1_a and lokr_w1_b should be provided")

        if (self.lokr_w2 is None) and ((self.lokr_w2_a is None) or (self.lokr_w2_b is None)):
            raise ValueError("Either lokr_w2 or both lokr_w2_a and lokr_w2_b should be provided")

        state_dict = {}

        if self.lokr_w1 is not None:
            state_dict[f"base_model.model.{self.peft_key}.lokr_w1"] = self.lokr_w1
        elif self.lokr_w1_a is not None:
            state_dict[f"base_model.model.{self.peft_key}.lokr_w1_a"] = self.lokr_w1_a
            state_dict[f"base_model.model.{self.peft_key}.lokr_w1_b"] = self.lokr_w1_b

        if self.lokr_w2 is not None:
            state_dict[f"base_model.model.{self.peft_key}.lokr_w2"] = self.lokr_w2
        elif self.lokr_w2_a is not None:
            state_dict[f"base_model.model.{self.peft_key}.lokr_w2_a"] = self.lokr_w2_a
            state_dict[f"base_model.model.{self.peft_key}.lokr_w2_b"] = self.lokr_w2_b

        if self.lokr_t2 is not None:
            state_dict[f"base_model.model.{self.peft_key}.lokr_t2"] = self.lokr_t2

        return state_dict

def construct_peft_loraconfig(info: Dict[str, LoRAInfo], **kwargs) -> LoraConfig:
    pass