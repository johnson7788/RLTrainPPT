#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/12 20:26
# @File  : load_model.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 使用proxy是不行的
# os.environ["ALL_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
from transformers import pipeline

# model = "unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit"
model = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
pipe = pipeline("text-generation", model=model)
messages = [
            {"role": "user", "content": "Who are you?"},
            ]
result = pipe(messages)
print(result)