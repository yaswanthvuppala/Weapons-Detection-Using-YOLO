# import os
# # Force CUDA debugging configuration
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from ultralytics import YOLO

def main():
    # Load a small YOLO model (highly recommended for 4GB VRAM)
    model = YOLO("yolo11s.pt")

    # Start training
    model.train(
        data=r"C:\Users\vuppa\Desktop\YOLO_Weapon_Detection\weapons\data.yaml",
        epochs=5,
        imgsz=640,
        batch=8,          # Try batch size 8 first (switch to 4 if you run out of memory)
        name="weapon_detector",
        patience=20,
        device=0,         # Run on RTX 3050 GPU
        workers=0,        # Safely uses 4 CPU threads to speed up dataloading on Windows
        plots=True        # Generates training metrics graphs
    )

if __name__ == '__main__':
    main()
