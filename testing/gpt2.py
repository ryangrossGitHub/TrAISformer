from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, TFGPT2LMHeadModel
from pathlib import Path
from tokenizers.implementations import ByteLevelBPETokenizer


paths = [str(x) for x in Path("./data/").glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()


# Customize training
tokenizer.train(files=paths)

# Save files to disk
tokenizer.save_model("./model")

tokenizer2 = GPT2Tokenizer.from_pretrained("./model")

tokenizer2.add_special_tokens({
  "eos_token": "</s>",
  "bos_token": "<s>",
  "unk_token": "<unk>",
  "pad_token": "<pad>",
  "mask_token": "<mask>"
})

config = GPT2Config(
  vocab_size=tokenizer2.vocab_size,
  bos_token_id=tokenizer2.bos_token_id,
  eos_token_id=tokenizer2.eos_token_id
)
# creating the model
model = TFGPT2LMHeadModel(config)

input_ids = tokenizer2.encode("8,7,", return_tensors='tf')

greedy_output = model.generate(input_ids, max_length=50)

print("Output:\n" + 100 * '-')
print(tokenizer2.decode(greedy_output[0], skip_special_tokens=False))