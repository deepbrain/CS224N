import os
try:
    token = os.environ["HF_API_TOKEN"]
except:
    token = None
    print('NO HF TOKEN FOUND. MAKE SURE YOU HAVE HF_API_TOKEN SET IN YOUR ENVIRONMENT')

if 'LIBRARY_ROOTS' in os.environ:
    del os.environ['LIBRARY_ROOTS']
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #unfortunately, this is necessary to define here to prevent the model from crashing when loading

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
import gc
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from peft import LoraConfig, get_peft_model, PeftModel
import torch
from loguru import logger
import asyncio
from functools import partial
from tokenized_dataset import TokenizedDataset, TokenizedQADataset
from evaluation import evaluate_on_nlp_tasks
BASE_PHI_REVISION = "accfee56d8988cae60915486310362db5831b1bd"
from numba import cuda


class MathLLM:
    def __init__(
        self,
        model_id,
        revision = None,
        use_quantization = False,
        use_vllm = False,
        load = False,
        max_context_length=2048,
        dataset_class=TokenizedDataset,
    ):
        self.model_id = model_id
        self.revision = revision
        self.use_quantization = use_quantization
        self.use_vllm = use_vllm
        self.model = None
        self.tokenizer = None
        self.HAS_BFLOAT16 = torch.cuda.is_bf16_supported()
        self.max_context_length = max_context_length
        self.dataset_class = dataset_class

        if self.HAS_BFLOAT16:
            self.torch_dtype = torch.float16 #need to match the Phi2 training type
        else:
            self.torch_dtype = torch.float16
        self.lock = asyncio.Lock()

        if load:
            self.model, self.tokenizer = self.load_model(model_id, revision)

    def load_base_model(self, model_id, revision):
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            load_in_8bit=False,
            return_dict=True,
            do_sample=True,
            device_map={'': 0},
#            attn_implementation="flash_attention_2",
            quantization_config=self.get_bnbs_config(),
            torch_dtype=self.torch_dtype,
        )

    def load_model(self, model_name=None, base_model_revision=None, use_vllm=None):
        self.unload_model()
        if use_vllm is not None:
            self.use_vllm = use_vllm
        if model_name is None:
            model_name = self.model_id
        if base_model_revision is None:
            base_model_revision = self.revision
        if model_name.endswith("-lora"):
            self.model, self.tokenizer = self.load_model_lora(model_name)
        elif self.use_vllm:
            self.model, self.tokenizer = self.load_model_vllm(model_name, base_model_revision)
        else:
            self.model, self.tokenizer = self.load_model_regular(model_name, base_model_revision)
        return self.model, self.tokenizer

    def unload_model(self):
        if self.model is not None:
            # free up memory
            if isinstance(self.model, LLM):
                del self.model
                gc.collect()
                torch.cuda.empty_cache()
            else:
                if hasattr(self, 'base_model') and self.base_model is not None:
                    del self.base_model
                    self.base_model = None
            del self.tokenizer
            torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None

    def load_model_vllm(self, model_name, base_model_revision):
        self.model = LLM(model=model_name, revision=base_model_revision)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=base_model_revision, trust_remote_code=True, use_fast=True)
        return self.model, self.tokenizer

    def process_batch_regular(self, batch, max_tokens, temperature = 0.2, top_p = 0.2, presence_penalty=1, frequency_penalty=1):
            # Tokenize the batch
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = self.tokenizer(batch, return_tensors="pt", padding='longest', truncation=True,
                                    pad_to_multiple_of=8).to(self.model.device)
            # Generate responses
            # Note: Adjust generation parameters as per your model's capabilities and requirements
            gen_args = {
                "max_length": inputs['input_ids'].shape[1] + max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "early_stopping": True, # Stop when EOS token is generated?
                "num_beams": 1, # "num_beams": 5, # Uncomment for beam search
                "pad_token_id": self.tokenizer.eos_token_id
            }
            if hasattr(self.model.config, 'presence_penalty'):
                gen_args['presence_penalty'] = presence_penalty
            if hasattr(self.model.config, 'frequency_penalty'):
                gen_args['frequency_penalty'] = frequency_penalty

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_args)

            # Decode the generated tokens
            decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            return decoded_outputs

    def process_batch(self, batch, max_tokens, temperature = 0.5, top_p = 0.2, presence_penalty=0.9, frequency_penalty=0.9):
           #processes a batch of prompts and returns a batch of solutions one per prompt
        # check if self.model is instance of the LLM class:
        if self.model is None:
            self.load_model(self.model_id, self.revision)
        if not isinstance(self.model, LLM):
            return self.process_batch_regular(batch, max_tokens, temperature, top_p, presence_penalty, frequency_penalty)
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty, max_tokens=max_tokens)
        results = self.model.generate(batch, sampling_params=sampling_params, use_tqdm=False)
        output = []
        for prompt, completion in zip(batch, results):
           output.append(prompt + completion.outputs[0].text)
        return output

    async def process_batch_async(self, batch, max_tokens, temperature=0.5, top_p=0.2, presence_penalty=0.1, frequency_penalty=0.1):
        async with self.lock:
            loop = asyncio.get_running_loop()
            # Use functools.partial to pass arguments to the function
            func = partial(self.process_batch, batch, max_tokens, temperature, top_p, presence_penalty, frequency_penalty)
            result = await loop.run_in_executor(None, func)
            return result


    def load_model_lora(self, model_id, adapter_only = False):
        peft_config = PeftConfig.from_pretrained(model_id)
        if not adapter_only: # load the base model too
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
            self.base_model = self.load_base_model(peft_config.base_model_name_or_path, self.revision)
        self.model = PeftModel.from_pretrained(
            self.base_model,
            model_id,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )
        self.model.eval()
        return self.model, self.tokenizer

    def load_model_regular(self, model_id, base_model_revision):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=base_model_revision, trust_remote_code=True, use_fast=True)
        self.model = self.load_base_model(model_id, base_model_revision)
        return self.model, self.tokenizer

    def get_bnbs_config(self):
        if self.use_quantization:
            compute_dtype = getattr(torch, "float16")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            return None

    def train(self, problems, eval_problems, new_model_id, lr = 1e-4, merge = False):
        # trains with a lora model on a batch of problems and saves the model to a new_model_id
        #  read the docs at https://huggingface.co/docs/trl/sft_trainer for more info

            if self.model_id.endswith("-lora"):
                peft_config = PeftConfig.from_pretrained(self.model_id)
                base_model_id = peft_config.base_model_name_or_path
                base_model_revision = self.revision
            else:
                base_model_id = self.model_id
                base_model_revision = self.revision

            self.unload_model()

            base_model = self.load_base_model(base_model_id, base_model_revision)
            base_tokenizer = AutoTokenizer.from_pretrained(base_model_id, revision=base_model_revision, use_fast=True)
            base_tokenizer.pad_token = base_tokenizer.eos_token
            base_tokenizer.padding_side = "right"


            train_data = self.dataset_class(problems, base_tokenizer, self.max_context_length)
            eval_data = self.dataset_class(eval_problems, base_tokenizer, self.max_context_length)


            base_model.config.use_cache = False
            base_model.config.eos_token_id = base_tokenizer.eos_token_id
            base_model.config.pretraining_tp = 1

            # gradient checkpointing to save memory
            base_model.gradient_checkpointing_enable()

            config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=[
                    'q_proj',
                    'k_proj',
                    'v_proj',
                    'dense',
                    'fc1',
                    'fc2',
                ],  # print(model) will show the modules to use
                bias="none",
                lora_dropout=0.1,
                task_type="CAUSAL_LM",
            )


            training_arguments = TrainingArguments(
                output_dir="logs",
                num_train_epochs=1,
                gradient_checkpointing=True, #------------
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=8,  # 4
                optim="paged_adamw_32bit",
                adam_beta1=0.9, #--------
                adam_beta2=0.95, #-----------
                save_steps=0,
                logging_steps=1,
                learning_rate=lr,
                weight_decay=0.01,
                fp16 = False,
                bf16 = False,
#                fp16=not self.HAS_BFLOAT16,
#                bf16=self.HAS_BFLOAT16,
                max_grad_norm=0.3,
                max_steps=-1,
                warmup_ratio=0.03,
                group_by_length=False,
                lr_scheduler_type="cosine",
                report_to="tensorboard",
                evaluation_strategy="epoch",
            )

            trainer = SFTTrainer(
                model=base_model,
                train_dataset=train_data,
                eval_dataset=eval_data,
                peft_config=config,
                tokenizer=base_tokenizer,
                args=training_arguments,
                packing=False,
                dataset_text_field="text",
                max_seq_length=self.max_context_length,
            )

            trainer.train()

            if not new_model_id.endswith("-lora"):
                new_model_id = new_model_id + "-lora"
            self.model_id = new_model_id
            self.tokenizer = base_tokenizer
            trainer.model.config.revision = base_model_revision
            trainer.model.save_pretrained(new_model_id)
            base_tokenizer.save_pretrained(new_model_id)
            self.model = trainer.model
            self.model.eval()
            if merge:
                if self.use_quantization: #TODO: implement merging in quantization mode
                    logger.warning("Merging is not supported in quantization mode")
                self.merge_lora()

    def evaluate(self):
        if isinstance(self.model, LLM):
            self.unload_model()
        if self.model is None:
            self.model, self.tokenizer = self.load_model_regular(self.model_id, self.revision)
        nlp_tasks_res = evaluate_on_nlp_tasks(self.model, self.revision, limit=300)
        logger.info(nlp_tasks_res["results"])
        del nlp_tasks_res
        self.unload_model()


    def merge_lora(self):
        # merges the lora model into its base model and saves it into a new model_id without the suffix -lora
        if not self.model_id.endswith("-lora"):
            logger.error("Model id does not end with -lora")
            return
        self.unload_model()
        new_model_id = self.model_id[:-5]
        self.model, self.tokenizer = self.load_model_lora(self.model_id)
        os.makedirs(new_model_id, exist_ok=True)
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(new_model_id)
        self.tokenizer.save_pretrained(new_model_id)
        del merged_model
        self.model_id = new_model_id
        self.unload_model()
#        self.load_model(new_model_id, self.revision)



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
from tqdm import tqdm
from datasets import Dataset
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

if 'LIBRARY_ROOTS' in os.environ:
    del os.environ['LIBRARY_ROOTS']

def generate_prompt(data_point):
    question = f"""The sentiment of the following phrase: '{data_point["text"]}' is
        \n\n Positive
        \n Negative
        \n Neutral
        \n Cannot be determined
        \n\nSolution: The correct option is """
    if 'sentiment' in data_point:
        answer = f"{data_point['sentiment']}"
        return question, answer
    return question

def load_data(dataset_class=TokenizedDataset):
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

    X_train = X_train.apply(generate_prompt, axis=1)
    X_eval = X_eval.apply(generate_prompt, axis=1)
    
    if dataset_class == TokenizedDataset:
        X_train = [q + a for q, a in X_train.tolist()]
        X_eval = [q + a for q, a in X_eval.tolist()]
        X_train = pd.DataFrame(X_train, columns=["text"])
        X_eval = pd.DataFrame(X_eval, columns=["text"])
        train_data = X_train['text'].tolist()
        eval_data = X_eval['text'].tolist()

    elif dataset_class == TokenizedQADataset:
        X_train = pd.DataFrame(X_train.tolist(), columns=["question", "answer"])
        X_eval = pd.DataFrame(X_eval.tolist(), columns=["question", "answer"])
        # return lists of (question, answer) pairs
        train_data = list(zip(X_train['question'].tolist(), X_train['answer'].tolist()))
        # return lists of (question, answer) pairs
        eval_data = list(zip(X_eval['question'].tolist(), X_eval['answer'].tolist()))

    y_true = X_test.sentiment
    X_test = pd.DataFrame(X_test[["text"]].apply(generate_prompt, axis=1), columns=["text"])

    return train_data, eval_data, X_test, y_true

@torch.no_grad()
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

@torch.no_grad()
def predict(X_test, LLM, batch_size=16):
    y_pred = []
    prompts = []
    max_len = len(X_test)
    for i in tqdm(range(max_len)):
        prompt = X_test.iloc[i]["text"]
        prompts.append(prompt)
        if (len(prompts) == batch_size) or (i == max_len - 1):
            results = LLM.process_batch(prompts, max_tokens=3)
            for result in results:
                answer = result.split("The correct option is")[-1].lower()
                if "positive" in answer:
                    y_pred.append("positive")
                elif "negative" in answer:
                    y_pred.append("negative")
                elif "neutral" in answer:
                    y_pred.append("neutral")
                else:
                    y_pred.append("none")
            prompts = []
    return y_pred

import logging

def main():
    training_dataset_class = TokenizedQADataset
    
    logger.add('math_llm.log', rotation="10 MB")
    logger.info(f"Using {training_dataset_class} for the training dataset.")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    train_data, eval_data, X_test, y_true = load_data(training_dataset_class)

#    logger.warning("DO NOT use quantization if your GPU has more than 10GB of memory. It is slow and can't be merged into base model.")
    LLM = MathLLM("microsoft/phi-2",  load=False, dataset_class=training_dataset_class)
    LLM.train(train_data, eval_data, "phi-2-pad2048", merge=True) # traning takes about 40 minutes on 1 A5000
    nlp_tasks_res = evaluate_on_nlp_tasks(LLM.model, LLM.revision, limit=10)
    logger.info(nlp_tasks_res["results"]) 
    del nlp_tasks_res
    LLM.unload_model()
    del LLM
    LLM = MathLLM("phi-2-pad2048-lora", load=True) #loading in lora mode
    y_pred = predict(X_test, LLM, batch_size=64)
    evaluate(y_true, y_pred)
    del LLM
    LLM = MathLLM("phi-2-pad2048", load=True) #loading in regular mode
    y_pred = predict(X_test, LLM, batch_size=64)
    evaluate(y_true, y_pred)
    del LLM
    LLM = MathLLM("phi-2-pad2048", use_vllm=True, load=True) #loading in vllm mode
    y_pred = predict(X_test, LLM, batch_size=64)
    evaluate(y_true, y_pred)



if __name__ == '__main__':
    main()