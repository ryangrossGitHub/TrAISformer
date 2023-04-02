from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./model",
    tokenizer="./model"
)

print(fill_mask("<s>8,7,<mask></s>"))