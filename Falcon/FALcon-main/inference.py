import torch
import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm

from FALcon_config_imagenet import FALcon_config
from FALcon_models_vgg import customizable_VGG as custom_vgg
from AVS_functions import extract_and_resize_glimpses_for_batch, get_grid, guess_TF_init_glimpses_for_batch

# -------------------- Configuration --------------------
config = FALcon_config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config.full_res_img_size = (224, 224)  # Resize CIFAR to ImageNet size

# -------------------- Load CIFAR-10 --------------------
transform = transforms.Compose([
    transforms.Resize(config.full_res_img_size),
    transforms.ToTensor(),
])

cifar_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(
    cifar_dataset, batch_size=config.batch_size_inf, shuffle=False, num_workers=2)

# -------------------- Load Model --------------------
model = custom_vgg(config).to(device)
checkpoint = torch.load(config.ckpt_dir, map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()

# -------------------- Extract Glimpses --------------------
output_dir = 'falcon_glimpses'
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    # The outer loop is per batch, not per image. 
    # All images in the batch go through the same processing logic simultaneously using batched tensors.
    for batch_idx, (images, _, _) in enumerate(tqdm(data_loader)):
        images = images.to(device)
        batch_size = images.size(0)

        # Use center box for dummy bbox
        H, W = config.full_res_img_size
        center_frac = 0.5
        center_w, center_h = int(W * center_frac), int(H * center_frac)
        x0, y0 = (W - center_w) // 2, (H - center_h) // 2
        dummy_bbox = torch.tensor([[x0, y0, center_w, center_h]], dtype=torch.float).repeat(batch_size, 1).to(device)

        glimpses_locs_dims = torch.zeros((batch_size, 4), dtype=torch.int).to(device)

        # Get center grid points for init glimpses
        all_grid_cells_centers = get_grid((W, H), config.glimpse_size_grid, grid_center_coords=True).to(device)
        init_glimpses_in_bbox = guess_TF_init_glimpses_for_batch(all_grid_cells_centers, dummy_bbox, is_inside_bbox=True)
        glimpses_centers = init_glimpses_in_bbox

        # Get glimpse locations and dimensions
        glimpses_locs_dims[:, 0] = glimpses_centers[:, 0] + 0.5 - (config.glimpse_size_grid[0] / 2.0)
        glimpses_locs_dims[:, 1] = glimpses_centers[:, 1] + 0.5 - (config.glimpse_size_grid[1] / 2.0)
        glimpses_locs_dims[:, 2] = config.glimpse_size_init[0]
        glimpses_locs_dims[:, 3] = config.glimpse_size_init[1]

        all_glimpses = []

        # Extract glimpses
        for g in range(config.num_glimpses):
            # Get glimpse locations and dimensions
            glimpses = extract_and_resize_glimpses_for_batch(
                images, glimpses_locs_dims,
                config.glimpse_size_fixed[1], config.glimpse_size_fixed[0])  # H, W order

            all_glimpses.append(glimpses.cpu())

            # Predict location/dimension changes
            glimpses_change_pred, switch_pred = model(glimpses)

            # Update glimpse regions
            probs = torch.sigmoid(glimpses_change_pred)
            actions = (probs >= config.glimpse_change_th)
            # torch.sigmoid is used to convert logits to probabilities using the sigmoid activation function
            
            
            x_min = glimpses_locs_dims[:, 0]
            x_max = x_min + glimpses_locs_dims[:, 2]
            y_min = glimpses_locs_dims[:, 1]
            y_max = y_min + glimpses_locs_dims[:, 3]

            # Update the coordinates of the glimpses based on the predicted actions
            x_min_new = torch.clamp(x_min - actions[:, 0]*config.glimpse_size_step[0], min=0)
            x_max_new = torch.clamp(x_max + actions[:, 1]*config.glimpse_size_step[0], max=W)
            y_min_new = torch.clamp(y_min - actions[:, 2]*config.glimpse_size_step[1], min=0)
            y_max_new = torch.clamp(y_max + actions[:, 3]*config.glimpse_size_step[1], max=H)
            # torch.clamp is used to ensure the new coordinates are within image bounds

            # Update the glimpse locations and dimensions
            glimpses_locs_dims[:, 0] = x_min_new
            glimpses_locs_dims[:, 1] = y_min_new
            glimpses_locs_dims[:, 2] = x_max_new - x_min_new
            glimpses_locs_dims[:, 3] = y_max_new - y_min_new

        # Stack all glimpses per image: [B, G, C, H, W]
        glimpse_tensor = torch.stack(all_glimpses, dim=1)  # shape: [B, G, C, H, W]

        # Save as .pt file
        torch.save(glimpse_tensor, os.path.join(output_dir, f'glimpses_batch{batch_idx:03d}.pt'))

