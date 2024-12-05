# loading a peft adapter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    OPTForCausalLM,
)
from peft import PeftConfig, LoraConfig
from transformers.trainer import Trainer

peft_model_id = "ybelkada/opt-350m-lora"
model = AutoModelForCausalLM.from_pretrained(peft_model_id)

# You can also load a PEFT adapter by calling the load_adapter method:

model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_id)
model.load_adapter(peft_model_id)

# loading the model in 8bit or 4bit

model = AutoModelForCausalLM.from_pretrained(
    peft_model_id, quantization_config=BitsAndBytesConfig(load_in_8bit=True)
)

# add a new adapter
model_2 = AutoModelForCausalLM.from_pretrained(model_id)

lora_config = LoraConfig(target_modules=["q_proj", "k_proj"], init_lora_weights=False)

model.add_adapter(lora_config, adapter_name="adapter_1")

# To add a new adapter:
model.add_adapter(lora_config, adapter_name="adapter_2")

# enabling and disabling adapters

# enable
adapter_model_id = "ybelkada/opt-350m-lora"
tokenizer = AutoTokenizer.from_pretrained(model_id)

text = "Hello, World!"
inputs = tokenizer(text, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(model_id)
peft_config = PeftConfig.from_pretrained(adapter_model_id)

peft_config.init_lora_weights = False

model.add_adapter(peft_config)
model.enable_paramters()
output = model.generate(**inputs)

# disable
model.disable_adapters()
output = model.generate(**inputs)


# train a PEFT Adapter

from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CASUAL_LM"
)
model.add_adapter(peft_config)

trainer = Trainer(model=model)
trainer.train()

model.save_pretrained("./")
model = AutoModelForCausalLM.from_pretrained("./")
