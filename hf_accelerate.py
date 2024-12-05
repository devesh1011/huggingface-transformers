from fine_tuning import train_loader, test_loader, model, optimizer, epochs, lr_scheduler, progress_bar
from accelerate import Accelerator

accelerator = Accelerator()

# prepare to accelerate
train_loader, test_loader, model, optimizer = accelerator.prepare(
    train_loader, test_loader, model, optimizer
)

# Backward
# The last addition is to replace the typical loss.backward() in your training loop with 🤗 Accelerate’s backward method

for epoch in range(epochs):
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# Train with a script
# If you are running your training from a script, run the following command to create and save a configuration file:

# Copied
# accelerate config
# Then launch your training with:

# Copied
# accelerate launch train.py

