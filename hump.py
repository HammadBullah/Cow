from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("/Users/hammadsafi/StudioProjects/MyApplication/COW/runs/detect/hump-detection2/weights/besthump.pt")

# Load image
image = cv2.imread("/Users/hammadsafi/StudioProjects/MyApplication/COW/foot/Foot-2/test/images/IMG_20210709_105647_1_jpg.rf.df215aa8b25bba18e5f40b8fb00e9e44.jpg")
output_image = image.copy()

# Constants
COW_PIXEL_HEIGHT = 3643
REAL_COW_HEIGHT_FT = 3
FT_PER_PIXEL = REAL_COW_HEIGHT_FT / COW_PIXEL_HEIGHT
PIXEL_TO_SQFT = FT_PER_PIXEL ** 2

# Run inference
results = model(image)[0]

cow_detected = False
hump_bbox = None
hump_area_sqft = None

# Draw boxes and calculate area
for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
    x1, y1, x2, y2 = map(int, box)
    width = x2 - x1
    height = y2 - y1
    area_pixels = width * height

    if int(cls) == 0:  # Cow
        cow_detected = True
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(output_image, "Cow", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    elif int(cls) == 1:  # Hump
        hump_bbox = (x1, y1, x2, y2)
        hump_area_sqft = area_pixels * PIXEL_TO_SQFT
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Hump: {hump_area_sqft:.4f} ft²"
        cv2.putText(output_image, label, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Final info
if cow_detected and hump_area_sqft:
    print(f"[RESULT] Hump area (bbox): {hump_area_sqft:.4f} ft² (cow present)")
elif hump_area_sqft and not cow_detected:
    print("[INFO] Hump detected, but cow not found — skipping area calc.")
else:
    print("[INFO] Hump not detected.")

# Show image
cv2.imshow("Hump Detection Overlay", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: save output image
cv2.imwrite("hump_detection_result.jpg", output_image)
