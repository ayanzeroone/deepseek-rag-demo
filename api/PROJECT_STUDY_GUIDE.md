# DeepSeek + Milvus RAG 项目学习文档（初学者版）

## 1. 项目在做什么
这个项目演示了一个最小可运行的 RAG（Retrieval-Augmented Generation，检索增强生成）流程：

1. 读取本地知识库（Milvus 文档的 Markdown 文件）。
2. 用 Embedding 模型把文本转成向量。
3. 把向量和原文写入 Milvus（本地 Milvus Lite）。
4. 用户提问时先做向量检索，找出最相关的文本片段。
5. 把检索结果拼到 Prompt 里，再交给 DeepSeek 生成答案。

一句话理解：先“找资料”，再“基于资料回答”。

---

## 2. 项目关键文件
- `deepseek_api.ipynb`：最小 DeepSeek API 连通性示例。
- `rag_milvus_deepseek.ipynb`：完整 RAG 主流程。
- `milvus_docs/`：知识库文档来源（Markdown）。
- `milvus_demo.db`：Milvus Lite 的本地数据库文件。
- `.milvus_demo.db.lock`：Milvus Lite 运行时锁文件。

---

## 3. RAG 主流程（按 notebook）

### 3.1 安装依赖
主要依赖：
- `pymilvus`：Milvus Python SDK。
- `milvus-lite`：本地向量库运行时（由 `pymilvus` 间接使用）。
- `openai`：用于调用 DeepSeek 的 OpenAI 兼容接口。
- `sentence-transformers`：本地 Embedding 模型。

### 3.2 准备 API Key
项目使用环境变量 `DEEPSEEK_API_KEY`。

### 3.3 加载文档
从 `milvus_docs/en/**/*.md` 读取文本，按 `# ` 粗分块。

### 3.4 初始化模型
- LLM：DeepSeek Chat（建议 `deepseek-v4-flash`）。
- Embedding：`SentenceTransformerEmbeddingFunction`（避免 `DefaultEmbeddingFunction` 的兼容问题）。

### 3.5 写入 Milvus
创建 collection，插入 `{id, vector, text}`。

### 3.6 查询与回答
- 问题向量化。
- 向量检索 top-k 片段。
- 拼接上下文，调用 DeepSeek 生成最终答案。

---

## 4. 推荐环境（稳定优先）
建议使用 `Python 3.11`，不要用 3.13 跑这个项目。

### 4.1 创建环境
```bash
conda create -n deepseek311 python=3.11 -y
conda activate deepseek311
python -m pip install -U pip setuptools wheel
```

### 4.2 安装依赖（固定版本）
```bash
python -m pip install "pymilvus[model]==2.5.10" "milvus-lite==2.4.11" "openai>=1.82.0" requests tqdm sentence-transformers ipykernel
```

> 不建议 `pip install -U ...` 全量升级，容易造成版本漂移。

### 4.3 注册 Jupyter 内核
```bash
python -m ipykernel install --user --name deepseek311 --display-name "Python (deepseek311)"
```

---

## 5. 初学者最容易踩的坑（本项目实战总结）

### 坑 1：`DefaultEmbeddingFunction()` 报 `encode_plus` 错
现象：
- `AttributeError: AlbertTokenizer has no attribute encode_plus`

原因：
- 默认 ONNX embedding 与当前 `transformers` 版本兼容性冲突。

做法：
- 改用：
```python
from pymilvus import model as milvus_model
embedding_model = milvus_model.dense.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu"
)
```

### 坑 2：`marshmallow` 版本冲突
现象：
- `AttributeError: module 'marshmallow' has no attribute '__version_info__'`

原因：
- 安装了 `marshmallow 4.x`，与 `pymilvus 2.5.10` 不匹配。

做法：
```bash
python -m pip install --force-reinstall --no-cache-dir "marshmallow<4,>=3.20.1" "pymilvus[model]==2.5.10"
```

### 坑 3：WSL 里无法下载 HuggingFace 模型
现象：
- `Network is unreachable` / `Failed to connect huggingface.co:443`

原因：
- WSL 进程未配置代理，或直连被网络策略限制。

做法：
```bash
export HF_ENDPOINT=https://hf-mirror.com
# 如需代理，按实际端口修改
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
```

### 坑 4：MilvusLite 数据库被锁
现象：
- `DataDirLockedError: another process holds the lock`

原因：
- 多个 notebook kernel/进程同时占用同一个 `milvus_demo.db`。

做法：
- 关闭重复 kernel；或换一个新 DB 文件名。
```python
milvus_client = MilvusClient(uri="./milvus_demo_2.db")
```

### 坑 5：`Search failed: function_score`
现象：
- 搜索时报 `function_score` 字段错误。

原因：
- `pymilvus` 与 `milvus-lite` 版本不匹配。

做法：
- 固定兼容版本（见第 4.2 节）。

---

## 6. 你可以直接套用的核心代码片段

### 6.1 DeepSeek 客户端
```python
import os
from openai import OpenAI

api_key = os.getenv("DEEPSEEK_API_KEY")
assert api_key, "DEEPSEEK_API_KEY 未设置"

deepseek_client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)
```

### 6.2 Embedding（推荐）
```python
from pymilvus import model as milvus_model

embedding_model = milvus_model.dense.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu"
)
```

### 6.3 读取文档（建议全量递归）
```python
from glob import glob

text_lines = []
for file_path in glob("milvus_docs/en/**/*.md", recursive=True):
    with open(file_path, "r", encoding="utf-8") as f:
        file_text = f.read()
    text_lines += [x.strip() for x in file_text.split("# ") if x.strip()]

print("chunks:", len(text_lines))
```

### 6.4 生成回答模型
```python
response = deepseek_client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
)
print(response.choices[0].message.content)
```

---

## 7. 初学者学习路线（建议）
1. 先跑 `deepseek_api.ipynb`，确保 API key 和模型调用通。
2. 再跑 `rag_milvus_deepseek.ipynb`，先验证 embedding 和 Milvus 建库。
3. 保证“检索结果可读”后，再调 Prompt。
4. 最后再做优化：
- 更好的 chunk 切分策略。
- 增加 reranker。
- 增加查询改写。
- 增加评估集与离线评测。

---

## 8. 这个项目学会后你掌握了什么
- LLM API 的基本调用。
- 向量化、向量数据库、语义检索的完整链路。
- RAG 的工程化最小闭环。
- 真实环境中的依赖、网络、锁冲突排障能力。

这就是从“能调用大模型”进阶到“能做知识问答系统”的关键一步。
