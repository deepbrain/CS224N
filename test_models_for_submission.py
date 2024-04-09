# test models for submission to hf

from transformers import AutoConfig, AutoModel, AutoTokenizer

def test_model(model_name, revision = None):
    config = AutoConfig.from_pretrained(model_name, revision=revision)
    print(config)
    model = AutoModel.from_pretrained(model_name, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)


if __name__ == '__main__':
    test_model("trained_iter_20240309-070712")
    print('done')