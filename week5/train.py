from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, WordPiece
from tokenizers.trainers import WordLevelTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace, BertPreTokenizer
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from config import get_weights_file_path, get_config

from tqdm import tqdm

from dataset import TranslationDataset, causal_mask
from model import build_transformer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split 



def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item[lang]

def get_or_build_tokenizer(config, dataset, lang):
    # "tokenizer_file": "tokenizer_{0}.json",
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)

        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_dataset(config):
    dataset_raw = load_dataset("bentrevett/multi30k")
    train_raw = dataset_raw['train']

    tokenizer_src = get_or_build_tokenizer(config, train_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, train_raw, config["lang_tgt"])

    train_dataset = TranslationDataset(train_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    validation_dataset = TranslationDataset(dataset_raw['validation'], tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    test_dataset = TranslationDataset(dataset_raw['test'], tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in train_raw:
        src_ids = tokenizer_src.encode(item[config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item[config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    # Max length of source sentence: 41
    # Max length of target sentence: 45

    train_dataloader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=True)

    return train_dataloader, validation_dataloader, tokenizer_src, tokenizer_tgt, test_dataloader

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], config["d_model"])
    return model

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break


    return decoder_input.squeeze(0)

def run_validation(model, validation_dataset, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples = 2):
    model.eval()

    count = 0
    source_texts = []
    expected_texts = []
    predicted_texts = []

    total_loss = 0
    num_batches = 0

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    console_width = 80

    with torch.no_grad():
        for batch in validation_dataset:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            decoder_input = batch['decoder_input'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            total_loss += loss.item()
            num_batches += 1
            
            assert encoder_input.size(0) ==1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected_texts.append(target_text)
            predicted_texts.append(model_out_text)

            if count <= num_examples:
                print_msg('-'*console_width)
                print_msg(f'Source: {source_text}')
                print_msg(f'Target: {target_text}')
                print_msg(f'Predicted: {model_out_text}')

    avg_val_loss = total_loss / num_batches
    return avg_val_loss
    
def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, validation_dataloader, tokenizer_src, tokenizer_tgt, _ = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    best_loss = float('inf')

    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
        best_loss = state.get("best_loss", float('inf'))

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch: 02d}")
        epoch_loss = 0
        num_batches = 0

        for batch in batch_iterator:
            model.train()
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch["label"].to(device)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"Loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

        avg_epoch_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch} Average Loss: {avg_epoch_loss:.4f}")

        avg_val_loss = run_validation(model, validation_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        print(f"Epoch {epoch} Average Validation Loss: {avg_val_loss:.4f}")

        writer.add_scalar('validation loss', avg_val_loss, global_step)
        writer.flush()

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_path = Path(config["model_folder"]) / "best.pt"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'best_loss': best_loss
            }, best_model_path)
            
            print(f"✓ New best model saved with loss: {best_loss:.4f}")

if __name__ == '__main__':
    config = get_config()
    ds = load_dataset("bentrevett/multi30k")
    print(ds["train"][0])

    train_model(config)

    


