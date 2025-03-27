import numpy as np
import cv2
import onnxruntime

# Vehicle classes (COCO format)
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def preprocess_image(image_path, input_size=(640, 640)):
    """Prepares the image for the model."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be read. Check the file path.")

    orig_img = img.copy()
    # Convert to the color space and size expected by the model
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img, orig_img, input_size

def process_detections(outputs, orig_img_shape, input_size=(640, 640), conf_threshold=0.5, iou_threshold=0.45):
    """
    Processes the model outputs and returns only vehicle detections.
    Scales the coordinates to match the original image size.
    """
    detection_output = outputs[0][0].T  # (8400, 84)
    boxes, scores, class_ids = [], [], []

    orig_h, orig_w = orig_img_shape[:2]
    input_h, input_w = input_size

    for det in detection_output:
        center_x, center_y, width, height = det[:4]
        class_confidences = det[4:84]

        class_id = np.argmax(class_confidences)
        score = class_confidences[class_id]

        if score < conf_threshold or class_id not in VEHICLE_CLASSES:
            continue

        # Scale coordinates to the original image size
        x1 = int((center_x - width / 2) * orig_w / input_w)
        y1 = int((center_y - height / 2) * orig_h / input_h)
        x2 = int((center_x + width / 2) * orig_w / input_w)
        y2 = int((center_y + height / 2) * orig_h / input_h)

        boxes.append([x1, y1, x2, y2])
        scores.append(float(score))
        class_ids.append(class_id)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)

    vehicle_detections = []
    for idx in indices.flatten():
        detection = {
            "class": VEHICLE_CLASSES[class_ids[idx]],
            "score": scores[idx],
            "bbox": boxes[idx]
        }
        # Debug: Print detection info
        print("Detection:", detection)
        vehicle_detections.append(detection)

    return vehicle_detections

def draw_detections(image, detections):
    """Draws the detected vehicles on the image."""
    for det in detections:
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f"{det['class'].capitalize()}: {det['score']:.2f}"

        # Debug: Print drawing coordinates
        print(f"Drawn box: ({x1}, {y1}), ({x2}, {y2}) - Label: {label}")

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def run_inference(image_path, model_path="yolov.onnx", output_path="output.jpg"):
    """Runs the model, detects vehicles, and saves the image."""
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    input_tensor, orig_img, input_size = preprocess_image(image_path)
    outputs = session.run(None, {input_name: input_tensor})

    # Get original image dimensions (for debugging)
    h_orig, w_orig = orig_img.shape[:2]
    print(f"Original image dimensions: width={w_orig}, height={h_orig}")

    detections = process_detections(outputs, orig_img.shape, input_size)

    if detections:
        result_img = draw_detections(orig_img, detections)
        cv2.imwrite(output_path, result_img)
        print(f"Drawing complete, output file: {output_path}")
    else:
        print("No vehicles detected, no drawing done.")

    return output_path

if __name__ == "__main__":
    image_path = "ar.jpg"   # Input image file path
    model_path = "yolov.onnx"  # ONNX model file path

    output_file = run_inference(image_path, model_path)
    print(f"Output file: {output_file}")
