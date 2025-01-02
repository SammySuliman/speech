from datasets import load_dataset
from transformers import MBart50Tokenizer, MBartForConditionalGeneration, Trainer, TrainingArguments

def finetune_mBART():

    ds = load_dataset("kaenakiakona/spanglish_claude_generated", split="train")

    splits = ds.train_test_split(test_size=0.2, seed=42)

    # Access the splits
    train_dataset = splits["train"]
    test_dataset = splits["test"]

    tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="es_ES")

    def preprocess_function(examples):
        # Tokenize inputs and outputs
        inputs = tokenizer(examples["context"], padding=True, truncation=True, max_length=512)
        targets = tokenizer(examples["response"], padding=True, truncation=True, max_length=512)
        
        # Add target input to the dictionary
        inputs["labels"] = targets["input_ids"]
        
        return inputs

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    train_dataset.set_format(type="torch", columns=['context', 'response', 'input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type="torch", columns=['context', 'response', 'input_ids', 'attention_mask', 'labels'])

    # Load mBART model
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    return tokenizer, model

if __name__ == '__main__':
    tokenizer, model = finetune_mBART()
    save_directory = "C:/Users/sammy/Desktop/Speech/"

    # Save model and tokenizer
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
