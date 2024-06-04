import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from tqdm import tqdm

import wandb
from model import GenerativePretrainedTransformer, get_positional_encoding

torch.manual_seed(1337)


# Get training data from github
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
def train_large_model():
    with open("data/input.txt") as f:
        raw_data = f.read()

    print(f"We have a total number of characters in this dataset of: {len(raw_data)}")

    train_data = raw_data[: int(0.9 * len(raw_data))]
    validation_data = raw_data[int(0.9 * len(raw_data)) :]
    # Define the tokenizer with encoding and decoding methods.
    all_unique_characters = set(train_data)
    n_unique_characters = len(all_unique_characters)
    print(
        f"The unique characters in the dataset are as follows:\n {''.join(sorted(all_unique_characters))}\n"
    )

    # We should add an <UNKOWN> token for characters not found in the training set
    all_unique_characters.add("<UNKOWN>")

    n_unique_characters = len(all_unique_characters)

    # Now we need a mapping from character to number and vice versa
    character_to_number_mapping = {
        c: i for i, c in enumerate(sorted(all_unique_characters))
    }
    number_to_character_mapping = {i: c for c, i in character_to_number_mapping.items()}

    class Tokenizer:
        def __init__(self, c_2_n_mapping, n_2_c_mapping):
            self.encoding_map = c_2_n_mapping
            self.decoding_map = n_2_c_mapping

        def encode(self, text):
            return [
                self.encoding_map.get(c, character_to_number_mapping["<UNKOWN>"])
                for c in text
            ]

        def decode(self, numbers):
            return "".join([self.decoding_map[n] for n in numbers])

    # Now let's create a tokenizer that can encode text and decode a sequence of numbers
    tokenizer = Tokenizer(character_to_number_mapping, number_to_character_mapping)

    test_string = "Hello World!"
    encoded_test_string = tokenizer.encode(test_string)
    decoded_test_string = tokenizer.decode(encoded_test_string)

    print(
        f"When encoding '{test_string}' with our tokenizer we get: {encoded_test_string}.\n"
    )
    print(
        f"Decoding the resulting sequence of numbers, we receive this: '{decoded_test_string}'."
    )

    EMBEDDING_SIZE = 384  # the vector dimensions
    N_TRANSFORMER_DECODER_BLOCKS = 6  # how many decoder blocks
    N_ATTENTION_HEADS = 6  # how many attention heads per decoder block
    MAXIMUM_SEQUENCE_LENGTH = (
        256  # how many characters the model should be able to handle
    )
    DROPOUT = 0.2  # regularization paramter to prevent overfitting

    BATCH_SIZE = 64  # how many sequences we pass in at once for training
    MAX_ITERATIONS = 2001
    DEVICE = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Whether to use the GPU or CPU

    compile = True

    torch.device(DEVICE)

    our_own_gpt = GenerativePretrainedTransformer(
        embedding_dimension=EMBEDDING_SIZE,
        n_attention_heads=N_ATTENTION_HEADS,
        maximum_sequence_length=MAXIMUM_SEQUENCE_LENGTH,
        n_transformer_blocks=N_TRANSFORMER_DECODER_BLOCKS,
        vocabulary_size=n_unique_characters,
    ).to(DEVICE)

    if compile:
        our_own_gpt = torch.compile(our_own_gpt)

    # Print the number of parameters in the model
    n_parameters = sum(p.numel() for p in our_own_gpt.parameters())
    print(f"Our model has {n_parameters/1e6:.2f}m parameters.")

    # This is the main training loop with batched data
    encoded_train_data = torch.tensor(tokenizer.encode(train_data), dtype=torch.long)
    encoded_validation_data = torch.tensor(
        tokenizer.encode(validation_data), dtype=torch.long
    )

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, data, sequence_length):
            self.data = data.to(DEVICE)
            self.sequence_length = sequence_length

        def __len__(self):
            return len(self.data) - self.sequence_length

        def __getitem__(self, idx):
            x = self.data[idx : idx + self.sequence_length]
            y = self.data[idx + 1 : idx + self.sequence_length + 1]
            return x, y

    train_dataset = TextDataset(encoded_train_data, MAXIMUM_SEQUENCE_LENGTH)
    validation_dataset = TextDataset(encoded_validation_data, MAXIMUM_SEQUENCE_LENGTH)

    # Now we can create the DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Logging preparation on wandb
    wandb.init(project="myGPT")

    # We will use the Adam optimizer and the CrossEntropyLoss
    # Define the optimizer and the loss function
    optimizer = torch.optim.AdamW(
        our_own_gpt.parameters(),
        lr=1e-3,
        betas=(0.9, 0.99),
        weight_decay=1e-1,
        fused=True,
    )
    criterion = nn.CrossEntropyLoss()

    # Adding simple way to reload best model based on validation loss
    best_validation_loss = float("inf")
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    iteration = 0
    validation_interval = 250
    log_interval = 10
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=MAX_ITERATIONS,
    )

    # Now we can start the training loop
    while iteration < MAX_ITERATIONS:
        our_own_gpt.train()
        for x, y in train_loader:
            if iteration >= MAX_ITERATIONS:
                break

            # Zero the gradientsls
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Forward pass
                logits = our_own_gpt(x)

                # Compute the loss
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            # Backward pass to update gradients of all parameters based on the loss
            # signaling how they should be adjusted to minimize the loss
            scaler.scale(loss).backward()
            # Update the weights/parameters accordingly
            scaler.step(optimizer)
            scaler.update()

            if (
                iteration % log_interval == 0
                and not iteration % validation_interval == 0
            ):
                print(f"Iteration: {iteration} - Training loss: {loss.item()}")

            # Logging the train loss
            wandb.log({"train_loss": loss})

            # After every validation_interval iterations we check for the validation set performance
            if iteration % validation_interval == 0 or iteration == MAX_ITERATIONS:
                our_own_gpt.eval()
                with torch.no_grad():
                    validation_loss = 0
                    for x, y in validation_loader:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            logits = our_own_gpt(x)
                            validation_loss += criterion(
                                logits.view(-1, logits.size(-1)), y.view(-1)
                            ).item()

                    validation_loss /= len(validation_loader)

                    # Logging val loss
                    wandb.log({"validation_loss": validation_loss})

                # If the model performs better, we save the parameters
                if validation_loss < best_validation_loss and iteration != 0:
                    best_validation_loss = validation_loss
                    print(
                        f"Saving the model dict with validation loss: {validation_loss}."
                    )
                    torch.save(our_own_gpt.state_dict(), "best_model.pth")

                print(
                    f"Iteration: {iteration} - Training loss: {loss.item()} - Validation loss: {validation_loss}"
                )

            iteration += 1
            scheduler.step()

    # Reload best model weights/parameters
    state_dict = torch.load("best_model.pth")
    our_own_gpt.load_state_dict(state_dict)

    # Log the model
    wandb.log_model(path="best_model.pth", name="large_model")
    wandb.finish()


if __name__ == "__main__":
    train_large_model()
