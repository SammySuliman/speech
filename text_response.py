# Function to load and process text from a file
def respond_to_text(text, tokenizer, model):
    try:
        # Tokenize the text (convert it to tokens)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=1024)

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Generate the response using the model
        outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)

        # Decode the token IDs back into text
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    except FileNotFoundError:
        print("File not found!")
        return None

