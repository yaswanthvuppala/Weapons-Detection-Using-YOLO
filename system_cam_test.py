from ultralytics import YOLO
import os
import cv2

# Path to your best trained model weights
MODEL_PATH = r"runs/detect/weapon_detector-5/weights/best.pt"

def main():
    # 1. Check if the model weights file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'.")
        print("Please verify the path to your best.pt weights.")
        return
        
    # 2. Load the trained weapon detector model
    print("Loading the weapon detection model...")
    model = YOLO(MODEL_PATH)
    
    print("\n" + "="*60)
    print("STARTING SYSTEM WEBCAM TEST")
    print("="*60)
    print("Instructions:")
    print("  1. A window showing your live webcam feed will open shortly.")
    print("  2. Bounding boxes will be drawn on detected 'person' and 'weapon' objects.")
    print("  3. To exit the webcam test, press 'q' on your keyboard.")
    print("="*60 + "\n")
    
    try:
        # Run prediction on the default system camera (source=0)
        #
        # Parameters explained:
        #  - source=0: Instructs YOLO to use your default system webcam.
        #  - show=False: We will handle showing the live window manually using cv2.
        #  - conf=0.30: Bounding boxes are only shown if confidence is above 30%.
        #  - stream=True: Processes frame-by-frame efficiently to prevent RAM usage build-up.
        # Force CPU execution to prevent CUDA crashes (e.g. illegal memory access or hangs in F.conv2d)
        results = model.predict(
            source=0,
            show=False,
            conf=0.30,
            stream=True,
            device='cpu'
        )
        
        # Loop through the stream generator and display each frame manually.
        for result in results:
            # Get the annotated frame
            annotated_frame = result.plot()
            
            # Display the annotated frame
            cv2.imshow("YOLO Weapon Detection - Webcam Feed", annotated_frame)
            
            # Press 'q' on the keyboard to exit the live feed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting webcam feed...")
                break
                
        # Clean up OpenCV windows
        cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check that:")
        print("  - Your webcam is connected.")
        print("  - No other app (like Teams, Zoom, or a browser) is currently using the camera.")

if __name__ == "__main__":
    main()
