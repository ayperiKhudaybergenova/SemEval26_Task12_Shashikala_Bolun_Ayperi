import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import torch.nn as nn
import torch.nn.functional as F

# ================= Focal Loss 定义 (保持不变) =================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (Tensor, optional): 类别权重
            gamma (float): 聚焦参数
            reduction (str): 'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is not None and self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss) 
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ================= 自定义 Trainer (保持不变) =================

class FocalLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None and self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)
            
        loss_fct = FocalLoss(alpha=self.class_weights, gamma=self.gamma)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# ================= 配置区域 =================
MODEL_NAME = "microsoft/deberta-v3-base"
OUTPUT_DIR = "./deberta_qwenRL_output_dropout"
MAX_LEN = 1024

# 基础文件路径
TRAIN_DOCS = "dataset/train_data_ranked/docs.json"
TRAIN_QS = "dataset/train_data_ranked/questions.jsonl"
VAL_DOCS = "dataset/val_data_ranked/docs.json"
VAL_QS = "dataset/val_data_ranked/questions.jsonl"

# === 集成模型输入路径 (Ensemble Input) ===
# 模型 A: Qwen
TRAIN_QWEN = "converted_json/train_results_492_qwen_converted.json"
VAL_QWEN = "converted_json/val_results_492_qwen_converted.json"

# 模型 B: Ministral (已修正文件名)
TRAIN_MINISTRAL = "converted_json/train_results_420_MINI_converted.json"
VAL_MINISTRAL = "converted_json/val_results_420_MINI_converted.json"


# ================= 数据预处理工具 =================

def load_docs_map(docs_path):
    print(f"Loading docs from {docs_path}...")
    with open(docs_path, 'r', encoding='utf-8') as f:
        docs_data = json.load(f)
    
    topic_map = {}
    for item in docs_data:
        t_id = item['topic_id']
        titles = [d.get('title', '') for d in item.get('docs', [])]
        titles_str = " | ".join([t for t in titles if t])
        topic_map[t_id] = titles_str
    return topic_map

def load_questions_map(qs_path):
    """
    建立 ID -> Question 的映射 (处理 q-201 和 201 的兼容性)
    """
    print(f"Loading questions from {qs_path}...")
    qs_map = {}
    with open(qs_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            raw_id = item.get("id") or item.get("index") or item.get("Question Index")
            
            if raw_id is not None:
                qs_map[raw_id] = item
                qs_map[str(raw_id)] = item
                
                # 兼容处理: 去掉 "q-" 前缀
                str_id = str(raw_id)
                if str_id.startswith("q-"):
                    pure_id = str_id.replace("q-", "")
                    qs_map[pure_id] = item
                    try:
                        qs_map[int(pure_id)] = item
                    except ValueError:
                        pass
                
    print(f"Loaded questions map size: {len(qs_map)}")
    return qs_map

def load_results_map(file_path):
    """
    加载模型输出并建立 ID -> Result 的映射。
    """
    print(f"Loading model results from {file_path}...")
    res_map = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for item in data:
            q_idx = item.get("Question Index")
            if q_idx is not None:
                res_map[q_idx] = item
                res_map[str(q_idx)] = item
    except FileNotFoundError:
        print(f"⚠️ Warning: File not found {file_path}")
        return {}
        
    print(f"Loaded {len(data)} results from file.")
    return res_map

# ================= Dataset 定义 (Ensemble 版本) =================

class EnsembleVerifierDataset(Dataset):
    def __init__(self, qwen_path, ministral_path, qs_path, docs_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        # 1. 加载所有资源
        topic_map = load_docs_map(docs_path)
        qs_map = load_questions_map(qs_path)
        
        # 2. 加载两个模型的输出
        qwen_map = load_results_map(qwen_path)
        ministral_map = load_results_map(ministral_path)
        
        matched_count = 0
        skipped_count = 0
        
        # 3. 遍历基准问题集
        seen_ids = set()
        unique_questions = []
        for q_item in qs_map.values():
            real_id = q_item.get("id") or q_item.get("index") or q_item.get("Question Index")
            if real_id not in seen_ids:
                seen_ids.add(real_id)
                unique_questions.append((real_id, q_item))

        print(f"Processing {len(unique_questions)} unique questions for ensemble...")

        for q_id, original_q in unique_questions:
            # 尝试从两个模型结果中获取数据
            qwen_item = qwen_map.get(q_id)
            ministral_item = ministral_map.get(q_id)

            if qwen_item is None and str(q_id).startswith("q-"):
                pure_id = str(q_id).replace("q-", "")
                qwen_item = qwen_map.get(pure_id) or qwen_map.get(int(pure_id) if pure_id.isdigit() else None)
            
            if ministral_item is None and str(q_id).startswith("q-"):
                pure_id = str(q_id).replace("q-", "")
                ministral_item = ministral_map.get(pure_id) or ministral_map.get(int(pure_id) if pure_id.isdigit() else None)

            if qwen_item is None and ministral_item is None:
                skipped_count += 1
                continue
            
            matched_count += 1
            
            # 获取基础信息
            topic_id = original_q.get("topic_id")
            target_event = original_q.get("target_event", "")
            doc_titles = topic_map.get(topic_id, "")
            
            golden_str = original_q.get("golden_answer", "")
            golden_set = set(x.strip() for x in golden_str.replace("，", ",").split(",") if x.strip())
            
            # 遍历4个选项生成样本
            for opt_key in ["A", "B", "C", "D"]:
                opt_text = original_q.get(f"option_{opt_key}", "")
                
                # --- 获取 Qwen 的判断与推理 ---
                reasoning_qwen = "No reasoning provided."
                verdict_qwen = "Unknown"
                if qwen_item and opt_key in qwen_item:
                    content = qwen_item[opt_key]
                    if isinstance(content, dict):
                        reasoning_qwen = content.get("reasoning", "No reasoning provided.")
                        # 重点：获取 is_correct 并转为字符串
                        verdict_qwen = str(content.get("is_correct", "Unknown"))
                
                # --- 获取 Ministral 的判断与推理 ---
                reasoning_ministral = "No reasoning provided."
                verdict_ministral = "Unknown"
                if ministral_item and opt_key in ministral_item:
                    content = ministral_item[opt_key]
                    if isinstance(content, dict):
                        reasoning_ministral = content.get("reasoning", "No reasoning provided.")
                        # 重点：获取 is_correct 并转为字符串
                        verdict_ministral = str(content.get("is_correct", "Unknown"))

                # --- 构建集成 Prompt (加入 Prediction 字段) ---
                # 格式: ... Qwen Prediction: True/False ...
                input_text = (
                    f"Event: {target_event} [SEP] "
                    f"Option: {opt_text} [SEP] "
                    f"Qwen Prediction: {verdict_qwen} | Logic: {reasoning_qwen} [SEP] "
                    #f"Ministral Prediction: {verdict_ministral} | Logic: {reasoning_ministral} [SEP] "
                    f"Ref_Titles: {doc_titles}"
                )
                
                label = 1 if opt_key in golden_set else 0
                
                self.samples.append({
                    "text": input_text,
                    "label": label
                })
                
        print(f"Stats: Processed {matched_count} questions. Skipped {skipped_count} (No model output found).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }

# ================= 评估指标 (官方 Metric) =================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    num_questions = len(labels) // 4
    reshaped_preds = predictions.reshape(num_questions, 4)
    reshaped_labels = labels.reshape(num_questions, 4)
    
    scores = []
    for i in range(num_questions):
        p_set = set()
        g_set = set()
        for j in range(4):
            if reshaped_preds[i, j] == 1: p_set.add(j)
            if reshaped_labels[i, j] == 1: g_set.add(j)
        
        if p_set == g_set:
            score = 1.0
        elif p_set.issubset(g_set) and len(p_set) > 0:
            score = 0.5
        else:
            score = 0.0
        scores.append(score)
    
    return {"official_score": np.mean(scores)}

# ================= 主程序 =================

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print(">>> Loading Training Set (Ensemble)...")
    train_dataset = EnsembleVerifierDataset(
        TRAIN_QWEN, TRAIN_MINISTRAL, TRAIN_QS, TRAIN_DOCS, tokenizer, max_len=MAX_LEN
    )
    
    # ================= 📏 第一招：修复后的长度检查代码 =================
    print("正在检查数据长度分布...")
    lengths = []
    
    # ❌ 错误写法: for item in train_dataset: (这会调用 __getitem__ 返回 Tensor)
    # ✅ 正确写法: 直接访问 dataset.samples (这里存着原始的 text)
    for item in train_dataset.samples:
        # item['text'] 是还没 tokenize 的长字符串
        # truncation=False 让 tokenizer 跑出完整长度
        tokens = tokenizer(item['text'], truncation=False, verbose=False)["input_ids"]
        lengths.append(len(tokens))
    
    lengths = np.array(lengths)
    print(f"最大长度: {lengths.max()}")
    print(f"平均长度: {lengths.mean()}")
    
    # 统计超过 1024 的比例
    overflow_ratio = (lengths > 1024).sum() / len(lengths)
    print(f"超过 1024 的比例: {overflow_ratio:.2%}")
    
    if overflow_ratio > 0.5:
        print("⚠️ 警告：超过一半的数据被截断了！模型可能看不到放在最后的 Logic 或 Docs！")
    else:
        print("✅ 数据长度健康，截断风险较低。")
    # ===============================================================

    print(">>> Loading Validation Set (Ensemble)...")
    val_dataset = EnsembleVerifierDataset(
        VAL_QWEN, VAL_MINISTRAL, VAL_QS, VAL_DOCS, tokenizer, max_len=MAX_LEN
    )

    if len(train_dataset) == 0:
        print("❌ Error: 训练集为空！请检查数据路径或ID匹配逻辑。")
        return
    '''
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2 
    )
    '''
    # 加载配置
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    
    # 增加 Dropout
    config.hidden_dropout_prob = 0.2         # 提高到 0.2 或 0.3
    config.attention_probs_dropout_prob = 0.2
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        config=config # 传入修改后的 config
    )
    
    if MAX_LEN > model.config.max_position_embeddings:
        print(f"⚠️ Extending model context window from {model.config.max_position_embeddings} to {MAX_LEN}...")
        model.config.max_position_embeddings = MAX_LEN
    '''
    # ================= ❄️ 冻结层 (防止过拟合核心) ❄️ =================
    # DeBERTa-v3-base 共有 12 层。
    # 策略：冻结 Embedding + 前 8 层，只训练最后 4 层和分类器。
    
    print("🥶 Freezing embeddings and bottom 8 layers...")
    
    # 1. 冻结 Embedding 层
    for param in model.deberta.embeddings.parameters():
        param.requires_grad = False
        
    # 2. 冻结前 8 层 Encoder (0-7)，只放开 8-11
    # model.deberta.encoder.layer 是一个 ModuleList
    for layer in model.deberta.encoder.layer[:8]:
        for param in layer.parameters():
            param.requires_grad = False
            
    # 验证一下：打印可训练参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Params: Total {total_params} | Trainable {trainable_params} ({trainable_params/total_params:.1%})")
    # =================================================================
    '''
    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # === 策略调整：松绑 + 高频抓拍 ===
        num_train_epochs=6,              # 没必要跑 10 轮了，反正前 3 轮定胜负
        learning_rate=2e-5,              # 【回升】：从 8e-6 回到 2e-5，恢复学习能力
        
        weight_decay=0.01,               # 【降低】：从 0.1 降回 0.01 (标准值)，别罚太重
        label_smoothing_factor=0.0,      # 【关闭】：必须关掉！我们需要模型绝对自信
        warmup_ratio=0.1,                # 正常预热
        
        # === 高频评估 (捕获 0.718 的关键) ===
        eval_strategy="steps",
        eval_steps=20,                   # 每 10 步 (约 20% Epoch) 就测一次
        save_strategy="steps",
        save_steps=20,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="official_score",
        greater_is_better=True,
        
        # === 显存控制 (配合 1024 长度) ===
        # 如果 1024 爆显存，把 batch 改为 2，accum 改为 16
        per_device_train_batch_size=2,   
        gradient_accumulation_steps=16,  
        
        fp16=True,
        logging_steps=5,
        report_to="none"
    )

    # 准备 Focal Loss 参数 (供 FocalLossTrainer 使用)
    current_class_weights = torch.tensor([2.0, 1.0], dtype=torch.float)
    current_gamma = 2.0
    
    # ================= Trainer 选择区域 =================
    
    # 【选项 A】使用自定义 Focal Loss (推荐，处理不平衡数据)
    '''
    print(f"Using FocalLossTrainer (Gamma={current_gamma}, Weights={current_class_weights})")
    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=current_class_weights, 
        gamma=current_gamma
    )
    '''

    # 【选项 B】使用标准 Cross Entropy (HuggingFace 默认)
    print("Using Standard Trainer (CrossEntropy)")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # ===================================================
    
    print(f"Starting Training...")
    trainer.train()
    
    save_path = os.path.join(OUTPUT_DIR, "final_ensemble_model")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Training Complete. Model saved to {save_path}")

if __name__ == "__main__":
    main()
