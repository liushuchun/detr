# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .detr_v2 import build_v2

def build_model(args):
    return build(args)

def build_model_v2(args):
    return build_v2(args)

