from pathlib import Path
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig, RobertaForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, \
    TrainingArguments, RobertaTokenizerFast, Trainer, pipeline, TFGPT2LMHeadModel

paths = [str(x) for x in Path("./data/").glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()


# Customize training
tokenizer.train(files=paths, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save_model("./model")

tokenizer2 = RobertaTokenizerFast.from_pretrained("./model", max_len=512)

tokenizer2._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

model = RobertaForMaskedLM(config=config)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer2,
    file_path="./data/train.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer2, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./out",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("./model")