from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
import torch.optim as optim
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

dataset = load_dataset("yelp_review_full")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

tokenized_datasets.set_format("torch")

# train dataset
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))

test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# creating dataloader
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-cased", num_labels=5
)

optimizer = optim.AdamW(model.parameters(), lr=5e-5)

epochs = 3
training_steps = epochs * len(train_loader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=training_steps,
)


if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
else:
    device = torch.device("cpu")


progress_bar = tqdm(range(training_steps))

model.train()
for epoch in range(epochs):
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


metric = evaluate.load("accuracy")
model.eval()
for batch in test_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
