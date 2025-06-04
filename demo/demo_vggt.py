import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from PIL import Image
import numpy as np

from romatch import roma_outdoor

from vggt.dependency.track_predict import predict_tracks
from vggt.utils.load_fn import load_and_preprocess_images_square


def _map_to_original(points, coord):
    """Map square coordinates back to the original image space."""
    x1, y1, x2, y2, width, height = coord
    scale_x = width / (x2 - x1)
    scale_y = height / (y2 - y1)
    pts = np.stack([
        (points[:, 0] - x1) * scale_x,
        (points[:, 1] - y1) * scale_y,
    ], axis=1)
    return pts


def match_with_vggt(im1_path, im2_path, device, query_points=1024):
    images, coords = load_and_preprocess_images_square([im1_path, im2_path], target_size=518)
    images = images.to(device)
    tracks, vis, *_ = predict_tracks(
        images,
        max_query_pts=query_points,
        query_frame_num=2,
        fine_tracking=False,
    )
    pts1 = _map_to_original(tracks[0], coords[0].cpu().numpy())
    pts2 = _map_to_original(tracks[1], coords[1].cpu().numpy())
    mask = vis[0] > 0.1
    return pts1[mask], pts2[mask]


def match_with_roma(im1_path, im2_path, device):
    model = roma_outdoor(device=device)
    warp, certainty = model.match(im1_path, im2_path, device=device)
    matches, _ = model.sample(warp, certainty)
    W_A, H_A = Image.open(im1_path).size
    W_B, H_B = Image.open(im2_path).size
    kptsA, kptsB = model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
    return kptsA.cpu().numpy(), kptsB.cpu().numpy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/toronto_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/toronto_B.jpg", type=str)
    parser.add_argument(
        "--save_dir",
        default="demo",
        type=str,
        help="Directory to save npz results",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    os.makedirs(args.save_dir, exist_ok=True)

    roma_k1, roma_k2 = match_with_roma(args.im_A_path, args.im_B_path, device)
    np.savez(os.path.join(args.save_dir, "roma_matches.npz"), kptsA=roma_k1, kptsB=roma_k2)
    torch.cuda.empty_cache()

    vggt_k1, vggt_k2 = match_with_vggt(args.im_A_path, args.im_B_path, device)
    np.savez(os.path.join(args.save_dir, "vggt_matches.npz"), kptsA=vggt_k1, kptsB=vggt_k2)

    print(f"RoMa matches: {len(roma_k1)}, VGGT matches: {len(vggt_k1)}")
