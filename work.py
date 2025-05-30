import cv2
import numpy as np
from ultralytics import YOLO
import os

# ---------- CONFIGURATION ----------
MODEL_PATH = "/Users/hammadsafi/StudioProjects/MyApplication/COW/fastcnn/best-2.pt"
INPUT_DIR = "/Users/hammadsafi/StudioProjects/MyApplication/COW/fastcnn/Foot-4/test/images"  # Folder containing input images
OUTPUT_DIR = "output"
MIN_CONTOUR_AREA = 50
REAL_COW_HEIGHT_FT = 3  # Real height of the cow in feet
PIXEL_TO_SQFT = None  # This will be calculated dynamically
CONF_THRESHOLD = 0.1  # Confidence threshold from your code
IOU_THRESHOLD = 0.1  # Increased IoU threshold for NMS to reduce duplicates

# ---------- CONSTANTS ----------
def calculate_ft_per_pixel(cow_bbox_height):
    """Calculate feet per pixel based on the detected cow's height in pixels."""
    ft_per_pixel = REAL_COW_HEIGHT_FT / cow_bbox_height
    print(f"[DETECTION] Cow pixel height: {cow_bbox_height}, ft/pixel: {ft_per_pixel:.4f}")
    return ft_per_pixel

def get_hoof_angle(hoof_crop, bbox_coords):
    if hoof_crop.shape[0] < 10 or hoof_crop.shape[1] < 10:
        print("[HOOF] Hoof crop too small:", hoof_crop.shape)
        return None, None, None

    gray = cv2.cvtColor(hoof_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary_mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
    if not valid_contours:
        print("[HOOF] No valid contours found")
        return None, None, None

    cnt = max(valid_contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = np.array(cv2.boxPoints(rect), dtype=np.float32)

    x1, y1, _, _ = bbox_coords
    box += [x1, y1]

    green_p1, green_p2 = box[1], box[2]
    if green_p1[1] > green_p2[1]:
        green_p1, green_p2 = green_p2, green_p1
    green_vec = green_p2 - green_p1

    green_lower = green_p2 if green_p2[1] > green_p1[1] else green_p1
    magenta_upper = box[3] if box[3][1] < box[0][1] else box[0]
    new_line_vec = magenta_upper - green_lower

    cos_theta = np.dot(green_vec, new_line_vec) / (np.linalg.norm(green_vec) * np.linalg.norm(new_line_vec))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta) * 180 / np.pi
    angle = min(angle, 180 - angle)

    return angle, rect, box

def draw_angle_overlay(image, angle, box):
    box = np.intp(box)
    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    cv2.line(image, tuple(box[0]), tuple(box[1]), (255, 0, 0), 2)
    cv2.line(image, tuple(box[1]), tuple(box[2]), (0, 255, 0), 2)
    cv2.line(image, tuple(box[2]), tuple(box[3]), (0, 255, 255), 2)
    cv2.line(image, tuple(box[3]), tuple(box[0]), (255, 0, 255), 2)

    green_lower = box[1] if box[1][1] > box[2][1] else box[2]
    magenta_upper = box[3] if box[3][1] < box[0][1] else box[0]
    cv2.line(image, tuple(green_lower), tuple(magenta_upper), (0, 255, 255), 2)

    x_text = int(min(box[:, 0]))
    y_text = int(min(box[:, 1]) - 10)
    cv2.putText(image, f"{angle:.1f} degrees", (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

def detect_objects_and_angle(image):
    global PIXEL_TO_SQFT  # Declare global at the start of the function
    try:
        model = YOLO(MODEL_PATH)
        print("[DETECTION] Model loaded successfully")
        print("[DETECTION] Model class names:", model.names)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return image

    results = model(image, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=True)[0]
    output_image = image.copy()

    cow_detected = False
    hump_detected = False
    hoof_detected = False
    FT_PER_PIXEL = None

    if len(results.boxes) == 0:
        print("[DETECTION] No objects detected by the model")
    else:
        print(f"[DETECTION] Found {len(results.boxes)} detections")

    # Collect cow and hump detections
    cow_detections = []
    hump_detections = []

    for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy(), results.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)
        class_name = model.names[class_id]
        print(f"[DETECTION] Class ID: {class_id}, Name: {class_name}, Confidence: {conf:.2f}, Bbox: [{x1}, {y1}, {x2}, {y2}]")

        if class_id == 0 or class_name.lower() == "cow":
            cow_detections.append((box, conf, class_name))
        elif class_id == 1 or class_name.lower() == "hump":
            hump_detections.append((box, conf, class_name))
        elif class_name.lower() == "hoof":
            hoof_detected = True
            hoof_crop = image[y1:y2, x1:x2]
            angle, rect, box_pts = get_hoof_angle(hoof_crop, (x1, y1, x2, y2))

            if angle is not None:
                draw_angle_overlay(output_image, angle, box_pts)
                print(f"[HOOF] Angle: {angle:.2f} degrees")
            else:
                print("[HOOF] Failed to calculate angle")
        else:
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (128, 128, 128), 1)
            cv2.putText(output_image, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    # Process the most confident cow detection first
    if cow_detections:
        top_cow = max(cow_detections, key=lambda x: x[1])
        x1, y1, x2, y2 = map(int, top_cow[0])
        conf = top_cow[1]
        class_name = top_cow[2]

        cow_detected = True
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(output_image, f"Cow ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Calculate feet per pixel based on cow height
        cow_bbox_height = y2 - y1
        FT_PER_PIXEL = calculate_ft_per_pixel(cow_bbox_height)
        PIXEL_TO_SQFT = FT_PER_PIXEL ** 2
        print(f"[DETECTION] Selected top cow detection with confidence: {conf:.2f}")
        print(f"[DETECTION] PIXEL_TO_SQFT set to: {PIXEL_TO_SQFT:.8f}")

    # Process hump detections after cow to ensure PIXEL_TO_SQFT is available
    for box, conf, class_name in hump_detections:
        x1, y1, x2, y2 = map(int, box)
        hump_detected = True
        hump_area_sqft = (x2 - x1) * (y2 - y1) * PIXEL_TO_SQFT if PIXEL_TO_SQFT else None
        label = f"Hump: {hump_area_sqft:.4f} sqft" if hump_area_sqft is not None else f"Hump ({conf:.2f})"
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_image, label, (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if cow_detected:
        print(f"[DETECTION] Cow detected, feet per pixel: {FT_PER_PIXEL:.4f} ft/pixel")
    else:
        print("[DETECTION] Cow not detected.")

    if hump_detected:
        if hump_area_sqft is not None:
            print(f"[DETECTION] Hump detected, area: {hump_area_sqft:.4f} sqft")
        else:
            print("[DETECTION] Hump detected, area: Not calculated (no cow detected)")
    else:
        print("[DETECTION] Hump not detected.")

    if hoof_detected:
        print(f"[DETECTION] Hoof detected.")
    else:
        print("[DETECTION] Hoof not detected.")

    return output_image

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png')

    # Get list of image files in INPUT_DIR
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)]
    if not image_files:
        print(f"❌ No images found in {INPUT_DIR}")
        return

    print(f"[INFO] Found {len(image_files)} images in {INPUT_DIR}")

    for image_file in image_files:
        image_path = os.path.join(INPUT_DIR, image_file)
        print(f"\n[INFO] Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            continue

        print("[DEBUG] Image loaded successfully, shape:", image.shape)

        # Detect cow, hump, and hoof with angle
        final_image = detect_objects_and_angle(image)

        # Save output
        output_filename = os.path.splitext(image_file)[0] + "_detected.jpg"
        out_path = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(out_path, final_image)
        print(f"✅ Output saved to: {out_path}")

    print("\n[INFO] All images processed.")

if __name__ == "__main__":
    main()