import os
import torch
from transformers import (
    AutoModelForQuestionAnswering, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments
)
from datasets import load_dataset

def prepare_train_features(examples, tokenizer, max_length=384, stride=128):
    """
    Prepare features for training
    """
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length'
    )
    
    # Prepare start and end positions
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        sequence_ids = tokenized_examples.sequence_ids(i)
        
        sample_index = sample_mapping[i]
        answer = examples['answers'][sample_index]
        
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        
        # Find token positions
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1
        
        # Detect answer span
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        tokenized_examples["start_positions"].append(token_start_index - 1)
        
        while offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        tokenized_examples["end_positions"].append(token_end_index + 1)
    
    return tokenized_examples

def train_model(
    dataset_path, 
    model_name='nlpaueb/legal-bert-base-uncased', 
    output_dir='./legal_bert_model'
):
    """
    Train the model on a custom dataset
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    # Load dataset (you need to prepare this)
    dataset = load_dataset('json', data_files=dataset_path)
    
    # Prepare dataset
    train_dataset = dataset['train'].map(
        lambda x: prepare_train_features(x, tokenizer), 
        batched=True, 
        remove_columns=dataset['train'].column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    # Example usage
    train_model('path/to/your/training_data.json')