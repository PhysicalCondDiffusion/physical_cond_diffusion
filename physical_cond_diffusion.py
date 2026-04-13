import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchmetrics.image import StructuralSimilarityIndexMeasure
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
import os
import json
from typing import Optional, Tuple
import pyvista as pv
import glob

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_name = "tmr_entities"
json_dir = "./tmr_dataset/annotations_" + dataset_name + ".json"
model_name = "./net_parameters_" + dataset_name + ".pth"

# configuration support image_size = 28, 64, 1024
image_size = 64
n_epochs = 80
learn_rate = 2e-4
batch_size = 64
NUM_WORKERS = 0
PIN_MEMORY = True
output_dir = "./output_images/" + dataset_name
carve_depth = 3  # setting from network output


class myDataset(Dataset):
    def __init__(self, image_id, data_image, data_tmr):
        self.image_ids = image_id
        self.image = data_image
        self.tmr = data_tmr

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = self.image[index]
        tmr = self.tmr[index]
        return image_id, image, tmr

    def __len__(self):
        return len(self.image)


class TMRDataset(Dataset):
    def __init__(self, tmr_data_list, image_json, transform=None):
        self.tmr_data_list = tmr_data_list
        self.image_json = image_json
        self.transform = transform
        self.image_map = {info["id"]: info for info in image_json}

    def __len__(self):
        return len(self.tmr_data_list)

    def __getitem__(self, idx):
        item = self.tmr_data_list[idx]
        image_id = item["image_id"]

        image_data = self.image_map.get(image_id)
        if image_data:
            img = Image.open(image_data["data_path"]).convert('L')
        else:
            raise FileNotFoundError(f"ID {image_id} error")

        if self.transform:
            img = self.transform(img)

        # --- Phase ---
        phase_path = item["tmr_phase_path"]
        phase_arr = np.loadtxt(phase_path, delimiter=',')

        phase_arr = phase_arr.astype(np.float32).copy()

        # Tensor: Shape (1, 10, 10)
        phase_tensor = torch.from_numpy(phase_arr).reshape(1, 10, 10)

        return image_id, img, phase_tensor


class ClassConditionedUnet(nn.Module):
    def __init__(self, class_emb_size=1):
        super().__init__()

        self.tmr_encoder = nn.Sequential(
            # Input: (Batch, 1, 10, 10) -> Only Phase
            nn.Conv2d(1, 32, kernel_size=3, padding=1),

            nn.BatchNorm2d(32),
            nn.SiLU(),
            # 10x10 -> 16x16
            nn.Upsample(scale_factor=1.6, mode='bilinear', align_corners=False),

            # 16x16 -> 32x32
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32

            # 32x32 -> 64x64
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 64

            nn.Conv2d(16, 4, kernel_size=3, padding=1)
        )
        in_channels = 1 + 4

        self.model = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x, t, class_labels):
        bs, ch, w, h = x.shape

        # Input: (bs, 1, 10, 10) -> Output: (bs, 4, 64, 64)
        class_cond = self.tmr_encoder(class_labels)

        if class_cond.shape[-1] != h:
            class_cond = F.interpolate(class_cond, size=(w, h), mode="bilinear", align_corners=False)
        net_input = torch.cat((x, class_cond), 1)
        return self.model(net_input, t).sample  # (bs, 1, w, h)


class DiffusionInference:
    def __init__(self, model_path, device='cuda', image_size=64):
        self.device = torch.device(device)
        self.image_size = image_size

        self.net = ClassConditionedUnet().to(self.device)
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval()

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='squaredcos_cap_v2'
        )
        self.inference_steps = 20
        self.noise_scheduler.set_timesteps(self.inference_steps)

    @torch.no_grad()
    def generate(self, phase_numpy):
        """
        phase_numpy (10x10)
        """
        d = phase_numpy.copy()

        phase_tensor = torch.from_numpy(d).float().view(1, 1, 10, 10).to(self.device)

        x = torch.randn(1, 1, self.image_size, self.image_size).to(self.device)

        for t in self.noise_scheduler.timesteps:
            with torch.autocast("cuda"):
                residual = self.net(x, t.to(self.device), phase_tensor)
            x = self.noise_scheduler.step(residual, t, x).prev_sample

        img_out = x.detach().cpu().squeeze().clip(-1, 1).numpy()
        return (img_out + 1) / 2.0


def evaluate_comprehensive_metrics(model, dataloader, scheduler, device, threshold=0.5):

    model.eval()
    mse_loss_fn = torch.nn.MSELoss()
    l1_loss_fn = torch.nn.L1Loss()  # MAE
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_mse = 0
    total_mae = 0
    total_ssim = 0

    total_iou = 0
    total_dice = 0
    total_precision = 0
    total_recall = 0

    count = 0
    smooth = 1e-6

    with torch.no_grad():
        for step, (img_ids, real_imgs, tmr_data) in enumerate(tqdm(dataloader)):
            real_imgs = real_imgs.to(device)
            tmr_data = tmr_data.to(device)
            batch_size = real_imgs.shape[0]

            x = torch.randn_like(real_imgs).to(device)
            for t in scheduler.timesteps:
                residual = model(x, t, tmr_data)
                x = scheduler.step(residual, t, x).prev_sample

            generated_imgs = (x.clamp(-1, 1) + 1) / 2

            pred_bin = (generated_imgs > threshold).float()

            target_bin = (real_imgs > threshold).float()

            loss_mse = mse_loss_fn(pred_bin, target_bin)
            loss_mae = l1_loss_fn(pred_bin, target_bin)
            score_ssim = ssim_metric(pred_bin, target_bin)

            total_mse += loss_mse.item() * batch_size
            total_mae += loss_mae.item() * batch_size
            total_ssim += score_ssim.item() * batch_size

            pred_flat = pred_bin.view(batch_size, -1)
            target_flat = target_bin.view(batch_size, -1)

            intersection = (pred_flat * target_flat).sum(1)
            pred_sum = pred_flat.sum(1)
            target_sum = target_flat.sum(1)

            # IoU
            union = pred_sum + target_sum - intersection
            iou = (intersection + smooth) / (union + smooth)

            # Dice
            dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)

            # Precision
            precision = (intersection + smooth) / (pred_sum + smooth)

            # Recall
            recall = (intersection + smooth) / (target_sum + smooth)

            # ç´?åŠ?
            total_iou += iou.sum().item()
            total_dice += dice.sum().item()
            total_precision += precision.sum().item()
            total_recall += recall.sum().item()

            count += batch_size

            # if step >= 2: break

    avg_mse = total_mse / count
    avg_mae = total_mae / count
    avg_ssim = total_ssim / count

    avg_iou = total_iou / count
    avg_dice = total_dice / count
    avg_precision = total_precision / count
    avg_recall = total_recall / count

    print(f"\n================ Final Evaluation Report ================")
    print(f"Total Test Samples: {count}")
    print("-" * 40)
    print(f"Image Similarity Metrics")
    print(f"  MSE             : {avg_mse:.6f} (Lower is better)")
    print(f"  MAE             : {avg_mae:.6f} (Lower is better)")
    print(f"  SSIM            : {avg_ssim:.4f} (Higher is better, max=1.0)")
    print("-" * 40)
    print(f"ã€Defect Segmentation Metricsã€?")
    print(f"  IoU             : {avg_iou:.4f} (Key Metric! >0.7 Excellent)")
    print(f"  Dice Score      : {avg_dice:.4f} (F1 Score)")
    print(f"  Precision       : {avg_precision:.4f}")
    print(f"  Recall          : {avg_recall:.4f}")
    print("==========================================================")

    model.train()

    return {
        "mse": avg_mse, "mae": avg_mae, "ssim": avg_ssim,
        "iou": avg_iou, "dice": avg_dice,
        "precision": avg_precision, "recall": avg_recall
    }


def save_image_plasma(img_tensor, save_path, dpi=150):

    img_np = img_tensor.detach().cpu().clip(-1, 1).numpy()

    if img_np.shape[0] == 3:
        img_np = img_np[0]  # (H, W)

    elif img_np.shape[0] == 1:
        img_np = img_np.squeeze(0)  # (H, W)

    plt.figure(figsize=(10, 8), dpi=dpi)
    plt.imshow(img_np, cmap='plasma')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()


def build_defect_volume(
        image_path: str,
        Lx: float = 35.0, Ly: float = 35.0, T: float = 5.0,
        nx: int = 200, ny: int = 200, nz: int = 25,
        carve_depth_mm: float = 1.5,
        thr: float = 0.50,
        base_color: str = "lightgrey",
        base_opacity: float = 0.2,
        defect_opacity: float = 1.0,
        defect_cmap: str = "viridis",
        background: str = "lightgrey",
        show_plot: bool = True,
        window_size: Tuple[int, int] = (1000, 700),
        screenshot_path: Optional[str] = None,
        return_plotter: bool = False,
        camera_pos: Optional[Tuple[Tuple, Tuple, Tuple]] = None,
):
    pv.global_theme.allow_empty_mesh = True

    sx, sy, sz = Lx / nx, Ly / ny, T / nz
    num_layers = int(carve_depth_mm / sz)

    img = Image.open(image_path).convert("L")
    img = img.resize((nx, ny), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

    values = np.zeros((nx, ny, nz), np.float32)
    processed_arr = np.flipud(arr).T
    for k in range(nz - num_layers, nz):
        values[:, :, k] = processed_arr

    grid = pv.ImageData()
    grid.dimensions = np.array(values.shape) + 1
    grid.spacing = (sx, sy, sz)
    grid.origin = (0.0, 0.0, 0.0)
    grid.cell_data["signal"] = values.ravel(order="F")

    defect_mesh = grid.threshold(thr, scalars="signal")

    if defect_mesh.n_points > 0:
        defect_mesh["thickness"] = defect_mesh.points[:, 2]
        vmin = T - carve_depth_mm
        vmax = T
    else:
        vmin, vmax = 0, 1

    p = None
    if show_plot or screenshot_path or return_plotter:
        p = pv.Plotter(window_size=window_size, off_screen=True)
        p.set_background(background)

        base_specimen = pv.Box(bounds=(0, Lx, 0, Ly, 0, T))
        p.add_mesh(
            base_specimen,
            color=base_color,
            opacity=base_opacity,
            smooth_shading=True
        )

        outline_box = pv.Box(bounds=(0, Lx, 0, Ly, 0, T))
        p.add_mesh(outline_box, color="white", style="wireframe", opacity=0.3)
        if defect_mesh.n_points > 0:
            p.add_mesh(
                defect_mesh,
                scalars="thickness",
                cmap=defect_cmap,
                clim=[vmin, vmax],
                opacity=defect_opacity,
                show_scalar_bar=False,
                lighting=True,
                specular=0.4
            )

        default_camera = [(70, -35, 50), (17.5, 17.5, 2.5), (0, 0, 1)]
        p.camera_position = camera_pos if camera_pos is not None else default_camera
        p.camera.zoom(1.2)

        if screenshot_path is not None:
            temp_img = p.screenshot()
            img_pil = Image.fromarray(temp_img).convert("RGBA")
            new_data = []
            for item in img_pil.getdata():
                if item[:3] == (211, 211, 211):
                    new_data.append((211,211,211,0))
                else:
                    new_data.append(item)
            img_pil.putdata(new_data)
            img_pil.save(screenshot_path, "PNG")
            print(f"save: {screenshot_path}")

        if show_plot:
            p.off_screen = False
            p.show()

    return grid, base_specimen, defect_mesh, (p if return_plotter else None)


if __name__ == '__main__':
    print(f'Using device: {device}')
    with open(json_dir) as f:
        train_json = json.load(f)
    # print(train_json)
    print(len(train_json["tmr_data"]))

    dataset_transformer = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    tmr_data = train_json["tmr_data"]
    image_info = train_json["image"]
    full_dataset = TMRDataset(tmr_data, image_info, transform=dataset_transformer)

    train_size = int(0.85 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    print(f"training size: {len(train_dataset)}, test size: {len(test_dataset)}")
    img_id, x, y = next(iter(train_dataloader))
    print('image_id', img_id)
    print('Input shape:', x.shape)
    print('tmr:', y)
    print('tmr shape:', y.shape)
    plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')
    plt.imshow(torchvision.utils.make_grid(y)[0], cmap='viridis')
    plt.show()

    # save_dir = "batch_y_phase_viridis"
    # os.makedirs(save_dir, exist_ok=True)

    yy = y.detach().cpu()

    if yy.dim() == 4:
        yy = yy[:, 0, :, :]
    elif yy.dim() == 3:
        pass
    else:
        raise ValueError(f"Unexpected y shape: {yy.shape}")

    for i in range(yy.size(0)):
        arr = yy[i].numpy()

        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)

        # plt.imsave(os.path.join(save_dir, f"y_phase_{i:03d}.png"), arr, cmap="viridis")
        # arr_upsampled = zoom(arr, zoom=20, order=1)
        # plt.imsave(os.path.join(save_dir, f"y_upsampled20x_{i:03d}.png"), arr_upsampled, cmap="viridis")

    print("Data show End!")

    # ******Train model (disable during loading)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    loss_fn = nn.MSELoss()
    net = ClassConditionedUnet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=learn_rate)
    losses = []
    for epoch in range(n_epochs):
        for _, x, y in tqdm(train_dataloader):
            x = x.to(device) * 2 - 1
            y = y.to(device)
            noise = torch.randn_like(x)
            T = noise_scheduler.num_train_timesteps
            timesteps = torch.randint(0, T, (x.shape[0],), device=device, dtype=torch.long).long()
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

            pred = net(noisy_x, timesteps, y)
            loss = loss_fn(pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        avg_loss = sum(losses[-100:]) / 100
        print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')
        if avg_loss < 0.002:
            break
    plt.plot(losses)
    # Save model (disable during loading)
    torch.save(net.state_dict(), model_name)

    step_save_dir = "./generation_steps"
    os.makedirs(step_save_dir, exist_ok=True)

    # # ******Load model (disable during training)
    # noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    # net = ClassConditionedUnet()
    # net.load_state_dict(torch.load(model_name))
    # net.to(device)
    # inference_steps = 50
    # noise_scheduler.set_timesteps(inference_steps)

    # # --- Evaluate model performance (may take time) ---
    # metrics = evaluate_comprehensive_metrics(net, test_dataloader, noise_scheduler, device)

    target_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 99]
    samples_per_id = 10
    total_samples = len(target_ids) * samples_per_id  # 10 * 10 = 100

    x = torch.randn(total_samples, 1, image_size, image_size).to(device)

    id_to_index_map = {
        test_dataset.dataset.tmr_data_list[idx]['image_id']: i
        for i, idx in enumerate(test_dataset.indices)
    }

    y_list = []
    print(f" image_id {target_ids} TMR...")

    for target_id in target_ids:
        if target_id in id_to_index_map:
            idx = id_to_index_map[target_id]

            _, _, tmr = test_dataset[idx]

            y_list.extend([tmr] * samples_per_id)
        else:
            print(f"Warning: image_id {target_id} not foundï¼?")
            y_list.extend([torch.zeros(1, 10, 10)] * samples_per_id)

    y = torch.stack(y_list).to(device)
    print(f"y shape: {y.shape}")

    print("Generating data...")
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
        with torch.no_grad():
            residual = net(x, t, y)

        x = noise_scheduler.step(residual, t, x).prev_sample

        grid_img = torchvision.utils.make_grid(
            x.detach().cpu().clip(-1, 1),
            nrow=samples_per_id,
            padding=2
        )
        step_img_path = os.path.join(step_save_dir, f"step_{i:02d}.png")
        save_image_plasma(grid_img, step_img_path)

    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    grid_img = torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=samples_per_id, padding=2)

    ax.imshow(grid_img[0], cmap='Greys_r')
    plt.title(f"Generated Images (Rows correspond to Image IDs: {target_ids})")
    plt.axis('off')
    plt.show()

    os.makedirs(output_dir, exist_ok=True)
    for idx, img in enumerate(x.detach().cpu().clip(-1, 1)):
        torchvision.utils.save_image(img, os.path.join(output_dir, f"sample_{idx}.jpg"))

    print(f"Generated process: {step_save_dir}")

    default_camera = (
        (52.5, -52.5, 15.0),
        (17.5, 17.5, 2.5),
        (0.0, 0.0, 1.0)
    )
    my_camera = (
        # (-80, 80, 80),
        # (17.5, 0, -100),
        (17.5, -60, 60),
        (17.5, 17.5, 2.5),
        (0, 0, 1)
    )

    save_dir = r"./output_3d_images"

    image_extensions = ["*.jpg"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(output_dir, ext)))

    for img_path in image_files:
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(save_dir, f"{file_name}.png")

        _grid, _mat_surf, _def_surf, _ = build_defect_volume(
            image_path=img_path,
            Lx=35.0, Ly=35.0, T=5.0,
            nx=200, ny=200, nz=25,
            carve_depth_mm=carve_depth,
            thr=0.7,
            defect_cmap="viridis",
            show_plot=True,
            return_plotter=False,
            camera_pos=my_camera,
            screenshot_path=save_path,
            screenshot_scale=2
        )

    print("\n Finishedï¼?")


