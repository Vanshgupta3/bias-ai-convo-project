import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

data = pd.read_csv("bias_dataset.csv")

labels = list(data["label"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    print(f"Epoch {epoch+1}")
    for text, label in zip(data["text"], data["label"]):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        label_id = torch.tensor([label2id[label]])
        outputs = model(**inputs, labels=label_id)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

model.save_pretrained("bias_model")
tokenizer.save_pretrained("bias_model")

print("MODEL TRAINED SUCCESSFULLY")
