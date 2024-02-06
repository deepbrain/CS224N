# this is a test how the Phi-2 model fine-tuning works, adapted from: https://github.com/nageshsinghc4/LLMs/blob/main/phi2_fine_tune_sentiment_analysis.ipynb
#download the data from: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news

#prevent the local variables from being imported into the remote environment as they can cuase crashes
import os
if 'LIBRARY_ROOTS' in os.environ:
    del os.environ['LIBRARY_ROOTS']
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
try:
    token = os.environ["HF_API_TOKEN"]
except:
    token = None
    print('NO HF TOKEN FOUND. MAKE SURE YOU HAVE HF_API_TOKEN SET IN YOUR ENVIRONMENT')

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import torch
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
                          logging)
import transformers
print(transformers.__version__)
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

from vllm import LLM, SamplingParams

from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate import Accelerator
from transformers import TrainingArguments, Trainer
from functools import partial
# from datasets import Dataset
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset



def generate_prompt(data_point):
    return f"""The sentiment of the following phrase: '{data_point["text"]}' is
            \n\n Positive
            \n Negative
            \n Neutral
            \n Cannot be determined
            \n\nSolution: The correct option is {data_point["sentiment"]}""".strip()

def generate_test_prompt(data_point):
    return f"""The sentiment of the following phrase: '{data_point["text"]}' is
            \n\n Positive
            \n Negative
            \n Neutral
            \n Cannot be determined
            \n\nSolution: The correct option is""".strip()




def load_data():
    filename = "./fin_sentiment_test_data/all-data.csv"

    df = pd.read_csv(filename,
                     names=["sentiment", "text"],
                     encoding="utf-8", encoding_errors="replace")
    X_train = list()
    X_test = list()
    for sentiment in ["positive", "neutral", "negative"]:
        train, test = train_test_split(df[df.sentiment == sentiment],
                                       train_size=300,
                                       test_size=300,
                                       random_state=42)
        X_train.append(train)
        X_test.append(test)

    X_train = pd.concat(X_train).sample(frac=1, random_state=10)
    X_test = pd.concat(X_test)

    eval_idx = [idx for idx in df.index if idx not in list(train.index) + list(test.index)]
    X_eval = df[df.index.isin(eval_idx)]
    X_eval = (X_eval
              .groupby('sentiment', group_keys=False)
              .apply(lambda x: x.sample(n=50, random_state=10, replace=True)))
    X_train = X_train.reset_index(drop=True)

    X_train = pd.DataFrame(X_train.apply(generate_prompt, axis=1),
                           columns=["text"])
    X_eval = pd.DataFrame(X_eval.apply(generate_prompt, axis=1),
                          columns=["text"])

    y_true = X_test.sentiment
    X_test = pd.DataFrame(X_test.apply(generate_test_prompt, axis=1), columns=["text"])

    #train_data = Dataset.from_pandas(X_train)
    #eval_data = Dataset.from_pandas(X_eval)
    return X_train, X_eval, X_test, y_true


class TokenizedDataset(Dataset):
    def __init__(self, list_of_strings, tokenizer, eval = False):
        self.data = []
        self.tokenizer = tokenizer
        self.total_calls = 0
        if eval:
            pad = "max_length"
            max_length = 2048
        else:
            pad = "do_not_pad"
            max_length = 2048
        for s in list_of_strings:
            encoded = tokenizer(
                text=s + tokenizer.eos_token,
                return_tensors="np",
                truncation=True,
                max_length=max_length,
                padding=pad,
            )
            self.data.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'labels': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.total_calls += 1
        return self.data[idx]


def tokenize_data(tokenizer, data):
    def collate_and_tokenize(example):
        prompt = example["text"]

        # Tokenize the prompt
        encoded = tokenizer(
            prompt,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            ## Very critical to keep max_length at 1024 on T4
            ## Anything more will lead to OOM on T4
            max_length=2048,
        )

        encoded["labels"] = encoded["input_ids"]
        return encoded

    tokenized_data = data.map(collate_and_tokenize, batched=True,  batch_size=1)
    return tokenized_data

def evaluate(y_true, y_pred):
    labels = ['positive', 'neutral', 'negative']
    mapping = {'positive': 2, 'neutral': 1, 'none':1, 'negative': 0}
    def map_func(x):
        return mapping.get(x, 1)

    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')

    # Generate accuracy report
    unique_labels = set(y_true)  # Get unique labels

    for label in unique_labels:
        label_indices = [i for i in range(len(y_true))
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')

    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print('\nClassification Report:')
    print(class_report)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])
    print('\nConfusion Matrix:')
    print(conf_matrix)


def load_model(model_name = "microsoft/phi-2", base_model_revision = "accfee56d8988cae60915486310362db5831b1bd"):
    model = LLM(model=model_name, revision=base_model_revision)

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=base_model_revision, trust_remote_code=True, use_fast=True)

    return model, tokenizer

def predict(X_test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(X_test))):
        prompt = X_test.iloc[i]["text"]
        sampling_params = SamplingParams(temperature=0.5, top_p=0.1, presence_penalty=0.1, frequency_penalty=0, max_tokens= 3)
        result = model.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
        answer = result[0].outputs[0].text.split("The correct option is")[-1].lower()
        if "positive" in answer:
            y_pred.append("positive")
        elif "negative" in answer:
            y_pred.append("negative")
        elif "neutral" in answer:
            y_pred.append("neutral")
        else:
            y_pred.append("none")
    return y_pred



def WebGLM_data(tokenizer):
    def collate_and_tokenize(examples):
        question = examples["question"][0].replace('"', r'\"')
        answer = examples["answer"][0].replace('"', r'\"')
        # unpacking the list of references and creating one string for reference
        references = '\n'.join([f"[{index + 1}] {string}" for index, string in enumerate(examples["references"][0])])

        # Merging into one prompt for tokenization and training
        prompt = f"""###System:
    Read the references provided and answer the corresponding question.
    ###References:
    {references}
    ###Question:
    {question}
    ###Answer:
    {answer}"""

        # Tokenize the prompt
        encoded = tokenizer(
            prompt,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            ## Very critical to keep max_length at 1024 on T4
            ## Anything more will lead to OOM on T4
            max_length=1024,
        )

        encoded["labels"] = encoded["input_ids"]
        return encoded

    from datasets import load_dataset

    # Load a slice of the WebGLM dataset for training and merge validation/test datasets
    train_dataset = load_dataset("THUDM/webglm-qa", split="train[5000:10000]")
    test_dataset = load_dataset("THUDM/webglm-qa", split="validation+test")

    columns_to_remove = ["question", "answer", "references"]

    # tokenize the training and test datasets
    tokenized_dataset_train = train_dataset.map(collate_and_tokenize,
                                                batched=True,
                                                batch_size=1,
                                                remove_columns=columns_to_remove)
    tokenized_dataset_test = test_dataset.map(collate_and_tokenize,
                                              batched=True,
                                              batch_size=1,
                                              remove_columns=columns_to_remove)
    return tokenized_dataset_train, tokenized_dataset_test


def count_unfrozen_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def read_lora(model_id):
    peft_config = PeftConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            load_in_8bit=False,
            return_dict=True,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        model,
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model

def merge_lora(lora_model, tokenizer, merged_model_id):
    #reads the lora trained model from the "model" folder and merges it with the original model
    #saves the merged model in the "lora_model" folder
    model = read_lora(lora_model)
    os.makedirs(merged_model_id, exist_ok=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_model_id)
    tokenizer.save_pretrained(merged_model_id)


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {'text' : self.df.iloc[idx]["text"]}

def train(trained_model_name, train_data, eval_data, base_model_id = "microsoft/phi-2", base_model_revision = "accfee56d8988cae60915486310362db5831b1bd", use_quantization = True):
    # expects dataset in form of dict with keys "text", read the docs at https://huggingface.co/docs/trl/sft_trainer for more info
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, revision=base_model_revision, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_data1 = TokenizedDataset(train_data["text"].tolist(), tokenizer)
    eval_data1 = TokenizedDataset(eval_data["text"].tolist(), tokenizer, eval = True)

    compute_dtype = getattr(torch, "float16")

    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(base_model_id,
                                                 revision=base_model_revision,
                                                 trust_remote_code=True,
                                                 quantization_config=bnb_config,
                                                 device_map="auto" )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
            'dense',
            'fc1',
            'fc2',
        ],  # print(model) will show the modules to use
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    HAS_BFLOAT16 = torch.cuda.is_bf16_supported()
    training_arguments = TrainingArguments(
        output_dir="logs",
        num_train_epochs=4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # 4
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16= not HAS_BFLOAT16,
        bf16=HAS_BFLOAT16,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        evaluation_strategy="epoch",
        eval_accumulation_steps = 5
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data1,
        eval_dataset=eval_data1,
        peft_config=config,
#        tokenizer=tokenizer,
        args=training_arguments,
        packing=True,
        max_seq_length=2048,
    )
    trainer.train()

    trainer.model.save_pretrained(trained_model_name)
    return trained_model_name, tokenizer


def train1(train_data, eval_data, model_name = "microsoft/phi-2"):
    # Configuration to load model in 4-bit quantized
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type='nf4',
                                   # bnb_4bit_compute_dtype='float16',
                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                    bnb_4bit_use_double_quant=True)

    # Loading Microsoft's Phi-2 model with compatible settings
    # Remove the attn_implementation = "flash_attention_2" below to run on T4 GPU
    # revision = "accfee56d8988cae60915486310362db5831b1bd"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto',
#                                                 quantization_config=bnb_config,
                                                 attn_implementation="flash_attention_2",
                                                 trust_remote_code=True)

    print("count of parameters in the model: ", count_unfrozen_params(model))

    # Setting up the tokenizer for Phi-2
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    tokenized_dataset_train, tokenized_dataset_test = WebGLM_data(tokenizer)


    #gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    # Freeze base model layers and cast layernorm in fp32
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    print(model)


    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
            'dense',
            'fc1',
            'fc2',
        ],  # print(model) will show the modules to use
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
            'dense',
            'fc1',
            'fc2',

        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    HAS_BFLOAT16 = False #torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir="phib",
        max_steps=100,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        warmup_steps=10,
        logging_steps=1,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=not HAS_BFLOAT16,
        bf16=HAS_BFLOAT16,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        group_by_length=True,
        # disable_tqdm=False,
        report_to="none",
        seed=3407,
    )

    #lora_model = get_peft_model(model, config)
    lora_model = get_peft_model(model, lora_config)
#    accelerator = Accelerator()
#    lora_model = accelerator.prepare_model(lora_model)

    print("count of parameters in the lora_model: ", count_unfrozen_params(lora_model))


    training_args1 = TrainingArguments(
        output_dir='./results',  # Output directory for checkpoints and predictions
        overwrite_output_dir=True,  # Overwrite the content of the output directory
        per_device_train_batch_size=2,  # Batch size for training
        per_device_eval_batch_size=2,  # Batch size for evaluation
        gradient_accumulation_steps=5,  # number of steps before optimizing
        gradient_checkpointing=True,  # Enable gradient checkpointing
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_steps=50,  # Number of warmup steps
        # max_steps=1000,  # Total number of training steps
        num_train_epochs=2,  # Number of training epochs
        learning_rate=5e-5,  # Learning rate
        weight_decay=0.01,  # Weight decay
        optim="paged_adamw_8bit",  # Keep the optimizer state and quantize it
        fp16=True,  # Use mixed precision training
        # For logging and saving
        logging_dir='./logs',
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,  # Limit the total number of checkpoints
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,  # Load the best model at the end of training
    )

    trainer = Trainer(
        model=lora_model,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_test,
        args=training_args,
    )

    model.config.use_cache = False

    start_time = time.time()
    trainer.train()
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time} seconds.")
    lora_model.push_to_hub("phi2-webglm-qlora",
                           use_auth_token=True,
                           commit_message="Training Phi-2",
                           private=True)

    new_model_name = "fin-trained-model"
    trainer.model.save_pretrained(new_model_name)
    return "fin-trained-model"

def main():
    train_data, eval_data, X_test, y_true = load_data()

    new_model_name, tokenizer = train("model2-test", train_data, eval_data)
    merge_lora(new_model_name, tokenizer, "lora-model")

    model, tokenizer = load_model("lora-model",None)

    y_pred = predict(X_test, model, tokenizer)
    evaluate(y_true, y_pred)


    y_pred = predict(X_test, model, tokenizer)



if __name__ == '__main__':
    main()