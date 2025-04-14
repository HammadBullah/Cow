import cv2
import numpy as np
from ultralytics import YOLO
import os

# ---------- CONFIGURATION ----------
MODEL_PATH = "/Users/hammadsafi/StudioProjects/MyApplication/COW/foot/train/weights/best.pt"
IMAGE_PATH = "/Users/hammadsafi/StudioProjects/MyApplication/COW/foot/Foot-2/test/images/IMG_20210709_105647_1_jpg.rf.df215aa8b25bba18e5f40b8fb00e9e44.jpg"
CLASS_NAME = "hoof"
MIN_CONTOUR_AREA = 50
OUTPUT_DIR = "output"
# -----------------------------------

def get_hoof_angle(hoof_crop, bbox_coords):
    """Estimate angle between green line (Side 2) and new connecting line."""
    if hoof_crop.shape[0] < 10 or hoof_crop.shape[1] < 10:
        print("Debug: Hoof crop too small.")
        return None, None, None

    # Preprocess
    gray = cv2.cvtColor(hoof_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary_mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Debug: Save binary mask
    cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_binary_mask.jpg"), binary_mask)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Debug: No contours found.")
        return None, None, None

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
    if not valid_contours:
        print(f"Debug: No contours with area > {MIN_CONTOUR_AREA}.")
        return None, None, None

    cnt = max(valid_contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)

    # Offset box to original image coordinates
    x1, y1, _, _ = bbox_coords
    box += [x1, y1]

    # Identify green line (Side 2: vertex 1 to vertex 2)
    green_p1 = box[1]
    green_p2 = box[2]
    if green_p1[1] > green_p2[1]:
        green_p1, green_p2 = green_p2, green_p1
    green_vec = green_p2 - green_p1

    # Find lower point of green line
    green_lower = green_p2 if green_p2[1] > green_p1[1] else green_p1

    # Find upper point of magenta line (Side 4: vertex 3 to vertex 0)
    magenta_upper = box[3] if box[3][1] < box[0][1] else box[0]

    # New connecting line vector (cyan line)
    new_line_vec = magenta_upper - green_lower

    # Calculate angle between green vector and new line vector
    cos_theta = np.dot(green_vec, new_line_vec) / (
        np.linalg.norm(green_vec) * np.linalg.norm(new_line_vec)
    )
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta) * 180 / np.pi
    # Ensure acute angle
    angle = min(angle, 180 - angle)

    print(f"Debug: Green line (Side 2): {green_p1} to {green_p2}")
    print(f"Debug: New line: {green_lower} to {magenta_upper}")
    print(f"Debug: Angle between green and new line: {angle:.2f}°")
    print(f"Debug: Vertices: 0={box[0]}, 1={box[1]}, 2={box[2]}, 3={box[3]}")

    return angle, rect, box

def draw_angle_overlay(image, rect, angle, bbox_coords, box):
    """Draw red rotated rectangle, colored sides, and new connecting line."""
    # Draw rotated rectangle (red)
    box = np.intp(box)
    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    # Draw each side with a different color
    cv2.line(image, tuple(box[0].astype(int)), tuple(box[1].astype(int)), (255, 0, 0), 2)  # Side 1 - Blue
    cv2.line(image, tuple(box[1].astype(int)), tuple(box[2].astype(int)), (0, 255, 0), 2)  # Side 2 - Green
    cv2.line(image, tuple(box[2].astype(int)), tuple(box[3].astype(int)), (0, 255, 255), 2)  # Side 3 - Yellow
    cv2.line(image, tuple(box[3].astype(int)), tuple(box[0].astype(int)), (255, 0, 255), 2)  # Side 4 - Magenta

    # Find lower point of green line (Side 2: vertex 1 to vertex 2)
    green_lower = box[1] if box[1][1] > box[2][1] else box[2]

    # Find upper point of magenta line (Side 4: vertex 3 to vertex 0)
    magenta_upper = box[3] if box[3][1] < box[0][1] else box[0]

    # Draw new line from lower point of green to upper point of magenta (cyan)
    cv2.line(image, tuple(green_lower.astype(int)), tuple(magenta_upper.astype(int)), (0, 255, 255), 2)

    # Draw angle text (yellow)
    x_text = int(min(box[:, 0]))
    y_text = int(min(box[:, 1]) - 10)
    cv2.putText(image, f"{angle:.1f}°", (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    print("Debug: Overlay drawn.")

def main():
    """Process image for hoof detection and angle estimation."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    try:
        model = YOLO(MODEL_PATH)
        print("Debug: Model loaded. Class names:", model.names)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Load image
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"❌ Error loading image: {IMAGE_PATH}")
        return
    height, width = image.shape[:2]
    print(f"Debug: Image size: {width}x{height}")

    # Inference
    results = model(IMAGE_PATH)[0]
    print(f"Debug: Total detections: {len(results.boxes)}")

    if len(results.boxes) == 0:
        print("⚠️ No hooves detected.")
        output_path = os.path.join(OUTPUT_DIR, f"hoof_{os.path.basename(IMAGE_PATH)}")
        cv2.imwrite(output_path, image)
        print(f"Debug: Saved image without annotations: {output_path}")
        return

    for i, box in enumerate(results.boxes):
        cls = int(box.cls[0])
        class_name = model.names[cls]
        conf = float(box.conf[0])
        print(f"Debug: Box {i + 1}: Class={class_name}, Confidence={conf:.2f}")

        if class_name.lower() == CLASS_NAME.lower():
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            print(f"Debug: Box coordinates: ({x1},{y1})-({x2},{y2})")

            if x2 <= x1 or y2 <= y1:
                print(f"⚠️ Invalid box {i + 1}. Skipping.")
                continue

            hoof_crop = image[y1:y2, x1:x2]
            debug_crop_path = os.path.join(OUTPUT_DIR, f"debug_crop_{i + 1}.jpg")
            cv2.imwrite(debug_crop_path, hoof_crop)
            print(f"Debug: Saved crop: {debug_crop_path}")

            angle, rect, box_points = get_hoof_angle(hoof_crop, (x1, y1, x2, y2))
            if angle is not None:
                print(f"✅ Hoof #{i + 1}: Confidence={conf:.2f}, Angle={angle:.2f}°")
                draw_angle_overlay(image, rect, angle, (x1, y1, x2, y2), box_points)
            else:
                print(f"⚠️ No angle computed for hoof #{i + 1}")

    # Save output
    output_path = os.path.join(OUTPUT_DIR, f"hoof_{os.path.basename(IMAGE_PATH)}")
    if cv2.imwrite(output_path, image):
        print(f"✅ Saved output: {output_path}")
    else:
        print(f"❌ Failed to save output: {output_path}")

    # Display image
    try:
        cv2.imshow("Hoof Angle Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("Debug: Display not supported.")

if __name__ == "__main__":
    main()