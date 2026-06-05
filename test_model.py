from ultralytics import YOLO
import os
import urllib.request

# 1. Define paths (referencing your trained model from weapon_detector-5)
MODEL_PATH = r"runs/detect/weapon_detector-5/weights/best.pt"
DATA_YAML = r"weapons/data.yaml"

def download_image(url, save_path):
    """
    Downloads an image from a URL to a local file path.
    Uses custom user-agent headers to avoid blocks.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            with open(save_path, 'wb') as out_file:
                out_file.write(response.read())
        return True
    except Exception as e:
        print(f"Failed to download image from URL. Error: {e}")
        return False

def test_on_test_data(model):
    """
    Evaluates the model on the test split defined in data.yaml
    and prints the performance metrics.
    """
    print("\n" + "="*65)
    print("1. EVALUATING MODEL ON TEST SPLIT")
    print("="*65)
    
    # Run validation on the test split
    metrics = model.val(
        data=DATA_YAML,
        split='test',  # Use 'test' split from data.yaml
        device=0,      # Use GPU if available
        plots=True     # Generate confusion matrix, PR curves, etc.
    )
    
    # Extract class names and results
    names = model.names
    
    print("\n--- CLASSIFICATION REPORT & METRICS ---")
    print(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'mAP50':<10} | {'mAP50-95':<10}")
    print("-" * 65)
    
    # Loop through each class and print individual metrics
    for i, name in names.items():
        # Precision, Recall, mAP50, mAP50-95 per class
        # Note: metrics.box.class_result(i) returns (precision, recall, ap50, ap)
        precision, recall, ap50, ap = metrics.box.class_result(i)
        print(f"{name:<15} | {precision:<10.4f} | {recall:<10.4f} | {ap50:<10.4f} | {ap:<10.4f}")
        
    print("-" * 65)
    
    # Print Mean metrics (across all classes)
    mean_precision = metrics.box.mp
    mean_recall = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    
    print(f"{'All (Mean)':<15} | {mean_precision:<10.4f} | {mean_recall:<10.4f} | {map50:<10.4f} | {map50_95:<10.4f}")
    print("="*65)
    print(f"Metrics saved to: {metrics.save_dir}")
    print(f"Check '{metrics.save_dir}/confusion_matrix.png' for the Confusion Matrix!")
    print("="*65 + "\n")


def predict_online_images(model, urls):
    """
    Tests the model on a list of online image URLs,
    downloading them first to avoid stream connection issues.
    """
    print("\n" + "="*65)
    print("2. TESTING ON RANDOM ONLINE IMAGES")
    print("="*65)
    
    temp_dir = "temp_downloads"
    os.makedirs(temp_dir, exist_ok=True)
    
    for idx, url in enumerate(urls, 1):
        print(f"\nProcessing Image {idx}: {url}")
        
        local_path = os.path.join(temp_dir, f"online_image_{idx}.jpg")
        
        if download_image(url, local_path):
            try:
                # Run prediction on the locally downloaded image file
                results = model.predict(
                    source=local_path,
                    save=True,       # Saves annotated images to runs/detect/predictX/
                    conf=0.25,       # Confidence threshold (adjust as needed)
                    device=0
                )
                
                for result in results:
                    path_saved = result.save_dir
                    print(f"Saved prediction results to folder: {path_saved}")
                    
                    # Print detected objects
                    if len(result.boxes) == 0:
                        print("No objects detected.")
                    else:
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            label = model.names[cls_id]
                            confidence = float(box.conf[0])
                            coords = box.xyxy[0].tolist() # [xmin, ymin, xmax, ymax]
                            print(f" - Detected: {label} (Confidence: {confidence:.2%}) at {coords}")
            except Exception as e:
                print(f"Failed to run prediction on downloaded image. Error: {e}")
            finally:
                # Cleanup individual temp file
                if os.path.exists(local_path):
                    try:
                        os.remove(local_path)
                    except Exception as cleanup_err:
                        pass
        else:
            print("Skipping prediction since download failed.")
            
    # Cleanup temp directory
    if os.path.exists(temp_dir):
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass
            
    print("="*65 + "\n")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}.")
        print("Please check the path to your trained model weights.")
        return
        
    print(f"Loading trained YOLO model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # Part 1: Test data evaluation
    test_on_test_data(model)
    
    # Part 2: Predict on random online images
    sample_online_urls = [
        # Image with security/police/weapons
        "https://images.unsplash.com/photo-1595590424283-b8f17842773f",
        # Random street scene image (testing for false positives)
        "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df"
    ]
    predict_online_images(model, sample_online_urls)

if __name__ == "__main__":
    main()
