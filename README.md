## fs-gpt

## 如何使用

### 环境安装

- 创建环境
```
conda create -n fs-gpt python=3.10
conda activate fs-gpt
```

- 安装项目
```
pip install -e ".[torch, embedding, inference]"
```

## 运行服务

### 词向量

```
fs-gpt run examples/embedding.yaml
```
### GPTQ量化

```
pip install -e ".[gptq]"
fs-gpt run examples/derive_gptq.yaml
```

## 开发计划

### 功能说明

- 预训练
- 微调训练：LoRA、QLoRA
- 模型合并：LoRA合并、量化、GPTQ、AWQ
- 推理接口：Bitsandbytes、vLLM、SGLang、LoRA切换
- 模型评估

## 相关参考

### 参考项目

- [xusenlinzy/api-for-open-llm](https://github.com/xusenlinzy/api-for-open-llm)
- [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

### 参考文档
