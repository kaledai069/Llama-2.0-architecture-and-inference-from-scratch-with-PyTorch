import torch
import time
import json

from pathlib import Path
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from typing import Optional

from model import ModelArgs, Transformer


class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()

        if load_model:
            checkpoint_paths = sorted(Path(checkpoints_dir).glob("*.pth"))

            assert len(checkpoint_paths) > 0, f"No checkpoint files found in the directory: {checkpoints_dir}"

            ckpt_path = checkpoint_paths[0]

            print(f"Loading Checkpoint from {ckpt_path} !!!")
            checkpoint = torch.load(ckpt_path, map_location = 'cpu')
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            

        # loading the parameters
        with open(Path(checkpoints_dir) / "params.json", 'r') as f:
            params = json.loads(f.read())

        # setting the model args required to estb the Transformer class
        model_args: ModelArgs = ModelArgs(
            max_seq_len = max_seq_len,
            max_batch_size = max_batch_size,
            device = device, 
            **params
        )

        # setup the tokenizer
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        # as per required by the Meta to set the default tensor type
        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        # instantiate the Transformer architecture with model argus without loading the pretrained params
        model = Transformer(model_args).to(device)

        prev_time = time.time()
        if load_model:
            del checkpoint['rope.freqs'] # cause this particular key wasn't implemented in this scratch code

            model.load_state_dict(checkpoint, strict = True) # strict means if any of the loaded checkpoint key doesn't match with architecture then it will raise an error / warning
            print(f"Loaded state dictionary from checkpoint into the model in {time.time() - prev_time:.2f}")

        return LLaMA(model, tokenizer, model_args)


if __name__ == '__main__':
    torch.manual_seed(0)

    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    model = LLaMA.build(
        checkpoints_dir = 'llama-2-7b/',
        tokenizer_path = 'tokenizer.model',
        load_model = True, 
        max_seq_len = 1024,
        max_batch_size = 3, 
        device = device
    )

    print("Yes, the pretrained model weights have been loaded !!!")