# %%
import torch
import torch.nn as nn

image_size = 64
batch_size = 32
channel = 3

# Create some data
data = torch.rand((batch_size, image_size, image_size, channel))

patch_size = 16
n_patches = (image_size // patch_size) ** 2
embedding_dimension = 768
embedder = nn.Conv2d(
    channel, out_channels=embedding_dimension, kernel_size=patch_size, stride=patch_size
)


embedded_data = embedder(data.permute(0, 3, 1, 2))
print(embedded_data.shape)
# Create a flattened sequence from this
embedded_data = embedded_data.flatten(2)
print(embedded_data.shape)
# Reorder the dimensions to (batch_size, n_patches, embedding_dimension)
embedded_data = embedded_data.transpose(1, 2)
print(embedded_data.shape)

# %%
import torch
import torch.nn as nn


class PatchAndProjectExplicit(nn.Module):
    def __init__(self, patch_size, hidden_dim, channels):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.channels = channels
        self.projection = nn.Linear(patch_size * patch_size * channels, hidden_dim)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        # Calculate the number of patches along each dimension
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        # Reshape to separate patches
        print(x.shape)
        x = x.view(
            batch_size,
            channels,
            num_patches_h,
            self.patch_size,
            num_patches_w,
            self.patch_size,
        )
        print(x.shape)
        # Permute to bring patches together and flatten the patch dimensions
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        print(x.shape)
        x = x.view(batch_size, num_patches_h * num_patches_w, -1)
        print(x.shape)

        # Project the patches
        x = self.projection(x)
        print(x.shape)
        return x


# Using nn.Conv2d for Patching and Projection
class Conv2dPatching(nn.Module):
    def __init__(self, patch_size, hidden_dim, channels):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        print(x.shape)
        x = self.conv(x)  # Conv2d will handle patching and projection
        print(x.shape)
        x = x.flatten(2)  # Flatten the spatial dimensions
        print(x.shape)
        x = x.transpose(
            1, 2
        )  # Adjust dimensions to match [batch, num_patches, hidden_dim]
        print(x.shape)
        return x


# Testing with the same dummy image and Conv2dPatching model for comparison
image = torch.rand((1, 3, 32, 32))  # Dummy input image
patch_size = 8  # 8x8 patches
hidden_dim = 64  # Projection dimension
channels = 3  # Image channels

# Initialize the models
explicit_model = PatchAndProjectExplicit(patch_size, hidden_dim, channels)
conv2d_model = Conv2dPatching(
    patch_size, hidden_dim, channels
)  # Assuming Conv2dPatching definition from earlier

# Forward pass
explicit_output = explicit_model(image)
conv2d_output = conv2d_model(image)

# Verify outputs
print("Output shapes are the same:", explicit_output.shape == conv2d_output.shape)
print(
    "Output difference (should be close to 0):",
    torch.norm(explicit_output - conv2d_output).item(),
)

# %%
