import cv2
import numpy as np
from ultralytics import YOLO
import os

# ---------- CONFIGURATION ----------
HUMP_MODEL_PATH = "/Users/hammadsafi/StudioProjects/MyApplication/COW/runs/detect/hump-detection2/weights/besthump.pt"
HOOF_MODEL_PATH = "/Users/hammadsafi/StudioProjects/MyApplication/COW/foot/train/weights/best.pt"
IMAGE_PATH = "/Users/hammadsafi/StudioProjects/MyApplication/COW/foot/Foot-2/test/images/IMG_20210709_105648_2_jpg.rf.b9d374543ae858f01620b7c853f87670.jpg"
OUTPUT_DIR = "output"
CLASS_NAME = "hoof"
MIN_CONTOUR_AREA = 50

REAL_COW_HEIGHT_FT = 3  # Real height of the cow in feet
PIXEL_TO_SQFT = None  # This will be calculated dynamically

# ---------- CONSTANTS ----------
def calculate_ft_per_pixel(cow_bbox_height):
    """Calculate feet per pixel based on the detected cow's height in pixels."""
    ft_per_pixel = REAL_COW_HEIGHT_FT / cow_bbox_height
    print("[HUMP] Cow pixel:", cow_bbox_height)
    return ft_per_pixel

def detect_hump(image):
    model = YOLO(HUMP_MODEL_PATH)
    results = model(image)[0]
    output_image = image.copy()

    cow_detected = False
    hump_area_sqft = None

    for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        width, height = x2 - x1, y2 - y1
        area_pixels = width * height

        if int(cls) == 0:  # Cow
            cow_detected = True
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(output_image, "Cow", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Use detected cow height in pixels to calculate feet per pixel
            cow_bbox_height = height
            global PIXEL_TO_SQFT
            FT_PER_PIXEL = calculate_ft_per_pixel(cow_bbox_height)
            PIXEL_TO_SQFT = FT_PER_PIXEL ** 2

        elif int(cls) == 1:  # Hump
            hump_detected= True
            hump_area_sqft = area_pixels * PIXEL_TO_SQFT if PIXEL_TO_SQFT else None
            label = f"Hump: {hump_area_sqft:.4f} sqft" if hump_area_sqft else "Hump detected"
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_image, label, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if cow_detected:
        print(f"[HUMP] Cow detected, feet per pixel: {FT_PER_PIXEL:.4f} ft/pixel")
    else:
        print("[HUMP] Cow not detected.")
    
    if hump_detected:
        print(f" HUMP detected, area: {hump_area_sqft:.4f} sqft")
    else:
        print("HUMP not detected.")

    return output_image

def get_hoof_angle(hoof_crop, bbox_coords):
    if hoof_crop.shape[0] < 10 or hoof_crop.shape[1] < 10:
        return None, None, None

    gray = cv2.cvtColor(hoof_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary_mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
    if not valid_contours:
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

def detect_hoof_and_angle(image):
    model = YOLO(HOOF_MODEL_PATH)
    results = model(image)[0]
    output_image = image.copy()

    for i, box in enumerate(results.boxes):
        cls = int(box.cls[0])
        class_name = model.names[cls]
        conf = float(box.conf[0])

        if class_name.lower() == CLASS_NAME.lower():
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            hoof_crop = image[y1:y2, x1:x2]
            angle, rect, box_pts = get_hoof_angle(hoof_crop, (x1, y1, x2, y2))

            if angle is not None:
                draw_angle_overlay(output_image, angle, box_pts)
                print(f"[HOOF] Angle: {angle:.2f} degrees")

    return output_image

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("❌ Failed to load image.")
        return

    # Detect hump and cow
    hump_image = detect_hump(image)

    # Detect hoof and angle
    final_image = detect_hoof_and_angle(hump_image)

    # Show final result
    cv2.imshow("Detection Result", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save output
    out_path = os.path.join(OUTPUT_DIR, "final_detection_result.jpg")
    cv2.imwrite(out_path, final_image)
    print(f"✅ Output saved to: {out_path}")

if __name__ == "__main__":
    main()
