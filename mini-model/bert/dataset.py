import numpy as np
import h5py
from torch.utils.data import Dataset
import torch

class pretraining_dataset_v1(Dataset):
    def __init__(self, f, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        keys = [
            "input_ids",
            "input_mask",
            "segment_ids",
            "masked_lm_positions",
            "masked_lm_ids",
            "next_sentence_labels",
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        print(f"Loaded {len(self.inputs[0]):d} samples from datafile: {input_file}")
        f.close()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs[0])

    def __getitem__(self, index):
        [
            input_ids,
            input_mask,
            segment_ids,
            masked_lm_positions,
            masked_lm_ids,
            next_sentence_labels,
        ] = [
            torch.from_numpy(input[index].astype(np.int64))
            if indice < 5
            else torch.from_numpy(np.asarray(input[index].astype(np.int64)))
            for indice, input in enumerate(self.inputs)
        ]
        masked_lm_labels = torch.zeros(input_ids.shape, dtype=torch.long) - 100
        index = self.max_pred_length
        masked_token_count = torch.count_nonzero(masked_lm_positions)
        if masked_token_count != 0:
            index = masked_token_count
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
        # print(f"input_mask_len = {torch.count_nonzero(input_ids)}  index = {index}")

        return [
            input_ids,
            segment_ids,
            input_mask,
            masked_lm_labels,
            next_sentence_labels,
        ]


class pretraining_dataset_v2(Dataset):
    def __init__(
        self, f, input_file, max_pred_length, max_seq_length=512, packed_samples=False
    ):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        self.max_seq_length = max_seq_length
        self.packed_samples = packed_samples

        if not self.packed_samples:
            keys = [
                "input_ids",
                "segment_ids",
                "masked_lm_positions",
                "masked_lm_ids",
                "next_sentence_labels",
            ]
        else:
            keys = [
                "input_ids",
                "segment_ids",
                "masked_lm_positions",
                "masked_lm_ids",
                "packed_input_len",
                "packed_masked_lm_len",
                "next_sentence_labels",
            ]

        self.inputs = [np.asarray(f[key][:]) for key in keys]
        print(f"Loaded {len(self.inputs[0]):d} samples from datafile: {input_file}")
        f.close()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs[0])

    def __getitem__(self, index):
        input_ids = np.zeros((self.max_seq_length)).astype(np.int64)
        input_mask = np.zeros((self.max_seq_length)).astype(np.int64)
        segment_ids = np.zeros((self.max_seq_length)).astype(np.int64)
        next_sentence_labels = np.zeros((3)).astype(np.int64)
        packed_input_len = np.zeros((3)).astype(np.int64)

        if not self.packed_samples:
            [
                _input_ids,
                _segment_ids,
                _masked_lm_positions,
                _masked_lm_ids,
                _next_sentence_labels,
            ] = [
                input[index].astype(np.int64)
                if indice < 4
                else np.asarray(input[index].astype(np.int64))
                for indice, input in enumerate(self.inputs)
            ]
        else:
            [
                _input_ids,
                _segment_ids,
                _masked_lm_positions,
                _masked_lm_ids,
                _packed_input_len,
                _packed_masked_lm_len,
                _next_sentence_labels,
            ] = [
                input[index].astype(np.int64)
                for indice, input in enumerate(self.inputs)
            ]

        input_mask_len = _input_ids.shape[-1]
        input_ids[:input_mask_len] = _input_ids
        input_mask[:input_mask_len] = np.ones((1, input_mask_len)).astype(np.int64)
        segment_ids[:input_mask_len] = _segment_ids
        masked_lm_labels = np.zeros(input_ids.shape, dtype=np.int64) - 100
        masked_lm_labels[_masked_lm_positions] = _masked_lm_ids

        if not self.packed_samples:
            next_sentence_labels = _next_sentence_labels

            return [
                torch.from_numpy(input_ids),
                torch.from_numpy(segment_ids),
                torch.from_numpy(input_mask),
                torch.from_numpy(masked_lm_labels),
                torch.from_numpy(next_sentence_labels),
            ]
        else:
            packed_seqs = _packed_input_len.shape[-1]
            next_sentence_labels[:packed_seqs] = _next_sentence_labels
            packed_input_len[:packed_seqs] = _packed_input_len

            return [
                torch.from_numpy(input_ids),
                torch.from_numpy(segment_ids),
                torch.from_numpy(input_mask),
                torch.from_numpy(masked_lm_labels),
                torch.from_numpy(next_sentence_labels),
                torch.from_numpy(packed_input_len),
            ]
            
def pretraining_dataset(
    input_file, max_pred_length, max_seq_length=512, packed_samples=False
):
    f = h5py.File(input_file, "r")
    if "input_mask" not in f.keys():
        return pretraining_dataset_v2(
            f, input_file, max_pred_length, max_seq_length, packed_samples
        )
    else:
        return pretraining_dataset_v1(f, input_file, max_pred_length)
