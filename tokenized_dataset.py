from loguru import logger
import numpy as np
from torch.utils.data import Dataset
import random

class TokenizedDataset(Dataset):
    def __init__(self, list_of_strings, tokenizer, max_length=2048):
        self.data = []
        self.tokenizer = tokenizer
        self.total_calls = 0
        self.total_length = 0
        tokenizer.padding_side = "right"
        pad = "do_not_pad"
        self.max_length = max_length
        for s in list_of_strings:
            encoded = tokenizer(
                text=s + tokenizer.eos_token,
                return_tensors="np",
                truncation=True,
                max_length=self.max_length,
                padding=pad,
            )
            self.total_length += encoded['input_ids'].shape[1]
            self.data.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'labels': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            })
        self.mean_length = self.total_length / len(list_of_strings)
        logger.info(f"Mean length of tokens per window: {self.mean_length}")
        self.pack(64)


    def pack(self, N):
        data_pack = self.data.copy()
        self.packed_data = []
        total_length = 0
        while data_pack:
            combined_item = {'input_ids': np.array([], dtype=np.int64), 'labels': np.array([], dtype=np.int64),
                             'attention_mask': np.array([], dtype=np.int64)}
            current_length = 0
            items_to_remove = []
            sample_size = min(N, len(data_pack))
            sampled_indices = random.sample(range(len(data_pack)), sample_size)
            random.shuffle(sampled_indices)  # Shuffle to ensure random pick order

            for idx in sampled_indices:
                item = data_pack[idx]
                item_length = len(item['input_ids'])
                if current_length + item_length <= self.max_length:
                    for key in combined_item:
                        combined_item[key] = np.concatenate([combined_item[key], item[key]]) if combined_item[key].size else item[key]
                    current_length += item_length
                    items_to_remove.append(idx)

            total_length += current_length
            # Padding to reach max_length - not sure why, but it seems to improve accuracy
            padding_length = self.max_length - current_length
            if padding_length > 0:
                pad_token_id = self.tokenizer.pad_token_id
                for key in combined_item:
                    combined_item[key] = np.pad(combined_item[key], (0, padding_length),
                                                constant_values=pad_token_id)

            # Ensure we always have at least one item to add to avoid empty data
            if items_to_remove:
                self.packed_data.append(combined_item)
                # Remove items from self.data in reverse order to avoid index issues
                for idx in sorted(items_to_remove, reverse=True):
                    del data_pack[idx]
            else:
                # Break the loop if no items were suitable to avoid infinite loop
                logger.warning("No items were suitable for packing, breaking loop")
                break
        logger.info(f"Mean length of tokens per packed window: {total_length / len(self.packed_data)}")
        self.total_calls = 0

    def __len__(self):
        return len(self.packed_data)

    def __getitem__(self, idx):
        self.total_calls += 1
        if self.total_calls > len(self.packed_data):
            prev_len = len(self.packed_data)
            self.pack(64)
            while len(self.packed_data) < prev_len:
                idx = random.randint(0, len(self.packed_data) - 1)
                self.packed_data.append(self.packed_data[idx])
        return self.packed_data[idx]
