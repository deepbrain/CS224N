import os
try:
    token = os.environ["HF_API_TOKEN"]
except:
    token = None
    print('NO HF TOKEN FOUND. MAKE SURE YOU HAVE HF_API_TOKEN SET IN YOUR ENVIRONMENT')
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
                          logging)
import transformers
print('Using transformers version: ' + transformers.__version__)
from vllm import LLM, SamplingParams

from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model, PeftModel
import torch
from loguru import logger
from torch.utils.data import Dataset

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
                return_tensors="np",  # Use PyTorch tensors
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

class MathLLM:
    def __init__(self, model_id, revision = "accfee56d8988cae60915486310362db5831b1bd"):
        self.model_id = model_id
        self.revision = revision
        self.model, self.tokenizer = self.load_model(model_id, revision)

    def load_model(self, model_name="microsoft/phi-2", base_model_revision="accfee56d8988cae60915486310362db5831b1bd"):
        if model_name.endswith("-lora"):
            return self.load_model_lora(model_name, base_model_revision)
        else:
            return self.load_model_vllm(model_name, base_model_revision)

    def load_model_vllm(self, model_name, base_model_revision):
        model = LLM(model=model_name, revision=base_model_revision)
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=base_model_revision, trust_remote_code=True, use_fast=True)
        return model, tokenizer

    def process_batch(self, batch, max_tokens, temperature = 0.5, top_p = 0.2, presence_penalty=0.1, frequency_penalty=0.1):
           #processes a batch of prompts and returns a batch of solutions one per prompt
        if self.model_id.endswith("-lora"):
            raise NotImplementedError #TODO implement this
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty, max_tokens=max_tokens)
        return self.model.generate(batch, sampling_params=sampling_params, use_tqdm=False)

    def load_model_lora(self, model_id, adapter_only = False, bnb_config=None):
        peft_config = PeftConfig.from_pretrained(model_id)
        if not adapter_only: # load the base model too
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                    peft_config.base_model_name_or_path,
                    load_in_8bit=False,
                    return_dict=True,
                    device_map="auto",
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
            )
        self.model = PeftModel.from_pretrained(
            self.base_model,
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        return self.model, self.tokenizer

    def train(self, problems, eval_problems, new_model_id, use_quantization = True, lr = 2e-4, merge = True, gpu_devices = '2,3'):
        # trains with a lora model on a batch of problems and saves the model to a new_model_id
        #  read the docs at https://huggingface.co/docs/trl/sft_trainer for more info
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

#            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision, use_fast=True)

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

            if not self.model_id.endswith("-lora"):
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id,
                                                             revision=self.revision,
                                                             trust_remote_code=True,
                                                             quantization_config=bnb_config,
                                                             device_map="auto")
            else:
                self.model, self.tokenizer = self.load_model_lora(self.model_id, bnb_config)

            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"

            train_data = TokenizedDataset(problems, self.tokenizer)
            eval_data = TokenizedDataset(eval_problems, self.tokenizer)


            self.model.config.use_cache = False
            self.model.config.pretraining_tp = 1

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
                fp16=not HAS_BFLOAT16,
                bf16=HAS_BFLOAT16,
                max_grad_norm=0.3,
                max_steps=-1,
                warmup_ratio=0.03,
                group_by_length=True,
                lr_scheduler_type="cosine",
                report_to="tensorboard",
                evaluation_strategy="epoch",
                eval_accumulation_steps=5
            )

            trainer = SFTTrainer(
                model=model,
                train_dataset=train_data1,
                eval_dataset=eval_data1,
                peft_config=config,
                dataset_text_field="text",
                tokenizer=tokenizer,
                args=training_arguments,
                packing=False,  # TODO debug code to work with packing - should be faster
                max_seq_length=2048,
            )
            trainer.train()

            self.model_id = new_model_id + "-lora"
            trainer.model.save_pretrained(new_model_id + "-lora")
            self.model = trainer.model
            if merge:
                self.merge_lora(new_model_id)


    def merge_lora(self):
        # remove suffix "-lora" from model_id
        if not self.model_id.endswith("-lora"):
            logger.error("Model id does not end with -lora")
            return
        new_model_id = self.model_id[:-5]
        model, tokenizer = self.load_model_lora(self.model_id)
        os.makedirs(new_model_id, exist_ok=True)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(new_model_id)
        self.tokenizer.save_pretrained(new_model_id)
        self.model_id = new_model_id
        self.model, self.tokenizer = self.load_model_vllm(new_model_id)



    def get_embeddings(self, problems): #returns embeddings for a batch of problems
        raise NotImplementedError

    def update(self, model_id): #load/update a model from model_id in the cloud
        raise NotImplementedError


##################################################################################
######################## UNIT TEST CODE ##########################################
##################################################################################

# this is a test how the Phi-2 model fine-tuning works, adapted from: https://github.com/nageshsinghc4/LLMs/blob/main/phi2_fine_tune_sentiment_analysis.ipynb
#download the data from: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
#prevent the local variables from being imported into the remote environment as they can cuase crashes
import pandas as pd
import numpy as np
import tqdm
from datasets import Dataset
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

if 'LIBRARY_ROOTS' in os.environ:
    del os.environ['LIBRARY_ROOTS']

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
                                       train_size=100,
                                       test_size=100,
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

    train_data = X_train['text'].tolist()
    eval_data = X_eval['text'].tolist()
    return train_data, eval_data, X_test, y_true


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

def predict(X_test, LLM):
    y_pred = []
    for i in tqdm(range(len(X_test))):
        prompt = X_test.iloc[i]["text"]
        result = LLM.process_batch(prompt, max_tokens=3)
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


def main():
    train_data, eval_data, X_test, y_true = load_data()
    LLM = MathLLM("microsoft/phi-2")
    LLM.train(train_data, eval_data, "phi-2-test")
    y_pred = predict(X_test, LLM)
    evaluate(y_true, y_pred)





if __name__ == '__main__':
    main()