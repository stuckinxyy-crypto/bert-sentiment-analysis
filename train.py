import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. 定义我们自己的数据集类
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 2. 定义评估指标计算函数
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # --- 数据加载与准备 ---
    print("Loading and preparing data...")
    # 加载数据
    df = pd.read_csv('data/ChnSentiCorp_htl_all.csv')
    df = df.dropna() # 去除缺失值
    
    # 划分数据集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['review'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )

    # --- Tokenizer 和模型加载 ---
    model_name = 'bert-base-chinese'
    print(f"Loading tokenizer and model for '{model_name}'...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # num_labels=2 因为是二分类（正面/负面）
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 创建数据集实例
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

    # --- 训练参数定义 ---
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir='./results',          # 输出目录
        num_train_epochs=3,              # 训练轮数
        per_device_train_batch_size=16,  # 训练时的batch size
        per_device_eval_batch_size=64,   # 验证时的batch size
        warmup_steps=500,                # 预热步数
        weight_decay=0.01,               # 权重衰减
        logging_dir='./logs',            # 日志目录
        logging_steps=10,
        evaluation_strategy="epoch",     # 每个epoch结束后进行评估
        save_strategy="epoch",           # 每个epoch结束后保存模型
        load_best_model_at_end=True,     # 训练结束后加载最优模型
        metric_for_best_model="f1",      # 以f1分数作为最优模型的标准
    )

    # --- Trainer 初始化与训练 ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    
    # --- 评估模型 ---
    print("Evaluating the best model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # --- 保存最终模型和tokenizer ---
    print("Saving the final model and tokenizer...")
    model.save_pretrained('./saved_model')
    tokenizer.save_pretrained('./saved_model')
    print("Training complete!")


if __name__ == "__main__":
    main()