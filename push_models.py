# (C) 2024 Stanford CS224N Group Custom Project by Artyom Shaposhnikov, Shubhra Mishra, Roberto Garcia
import huggingface_hub
import os
try:
    token = os.environ["HF_API_TOKEN"]
except:
    token = None
    print('NO HF TOKEN FOUND. MAKE SURE YOU HAVE HF_API_TOKEN SET IN YOUR ENVIRONMENT')
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


from math_llm import MathLLM

def upload():
    huggingface_hub.logout()
    huggingface_hub.login(token=token)
    model = MathLLM("trained_iter_20240214-181649", use_vllm=False, load=True)
    model.push_to_hub("phi2-gsm8k-cross-prompt-training")
    del model
    model = MathLLM("trained_iter_20240215-134533", use_vllm=False, load=True)
    model.push_to_hub("phi2-gsm8k-single-prompt-temperature-training")
    del model
    model = MathLLM("trained_iter_20240309-070712", use_vllm=False, load=True)
    model.push_to_hub("phi2-gsm8k-rephrase-high-confidence-training")


if __name__ == '__main__':
    upload()
    print('done')