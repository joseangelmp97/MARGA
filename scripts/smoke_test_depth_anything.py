# scripts/smoke_test_depth_anything.py
import cv2

from src.core.types import FramePacket
from src.perception.depth_anything_v2 import DepthAnythingV2Config, DepthAnythingV2Estimator

IMG_PATH = "./assets/test1.jpg"  # pon aquí una imagen real


def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise RuntimeError(f"Could not read image: {IMG_PATH}")

    packet = FramePacket.from_frame(img)

    cfg = DepthAnythingV2Config(
        repo_dir="models/Depth-Anything-V2",
        encoder="vits",
        checkpoint_path="models/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth",
        device="cpu",
    )

    est = DepthAnythingV2Estimator(cfg)
    # Mostrar mapa de calor con ROI (bloqueante hasta que pulses una tecla)
    est.visualize_depth(packet, block=True)
    score = est.estimate_roi_depth(packet)
    print("ROI depth score:", score)
    est.close()


if __name__ == "__main__":
    main()
