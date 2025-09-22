import os
import json
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
# 1. Path to the directory containing your input YOLO .txt files
INPUT_DIR = "../datasets/train/labels"

# 2. Path to the directory containing your corresponding image files
IMAGE_DIR = "../datasets/train/images"

# 3. Path to the directory where the output .json files will be saved
OUTPUT_DIR = "output_json"

# 4. The file extension of your images (e.g., .jpg, .png)
IMAGE_EXTENSION = ".jpg"

# 5. !! IMPORTANT !!
#    You MUST update this dictionary to map your class numbers (from the .txt files)
#    to the corresponding category names.
CLASS_MAPPING = {
    1: "person",
    2: "object1",
    3: "car",
    4: "motorcycle",
    5: "truck",
    6: "bycycle",
    7: "bus",
    8: "object2",
    9: "object3"
    # Add all your class mappings here...
}


# ---------------------


def yolo_to_box2d(x_center, y_center, width, height, img_width, img_height):
    """
    Converts YOLO format to BDD100K box2d format.
    """
    abs_width = width * img_width
    abs_height = height * img_height
    abs_x_center = x_center * img_width
    abs_y_center = y_center * img_height

    x1 = abs_x_center - (abs_width / 2)
    y1 = abs_y_center - (abs_height / 2)
    x2 = abs_x_center + (abs_width / 2)
    y2 = abs_y_center + (abs_height / 2)

    return x1, y1, x2, y2


def convert_yolo_to_bdd_frames():
    """
    Converts YOLO annotations to individual BDD100K-style JSON files (one per image),
    matching the detailed frame/object structure.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Annotations source: '{INPUT_DIR}'")
    print(f"Images source:      '{IMAGE_DIR}'")
    print(f"JSON output:        '{OUTPUT_DIR}'")

    try:
        annotation_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
        if not annotation_files:
            print(f"Error: No .txt files found in '{INPUT_DIR}'.")
            return
    except FileNotFoundError:
        print(f"Error: Directory '{INPUT_DIR}' not found.")
        return

    print(f"\nFound {len(annotation_files)} annotation files. Starting conversion...")

    for filename in tqdm(annotation_files, desc="Converting files"):
        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(IMAGE_DIR, base_name + IMAGE_EXTENSION)

        if not os.path.exists(image_path):
            continue

        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception:
            continue

        # This list will hold all object dictionaries for the current image
        objects_list = []

        txt_path = os.path.join(INPUT_DIR, filename)
        with open(txt_path, 'r') as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                try:
                    class_no = int(parts[0])
                    x_center, y_center, norm_w, norm_h = map(float, parts[1:])
                except ValueError:
                    continue

                if class_no not in CLASS_MAPPING:
                    continue

                x1, y1, x2, y2 = yolo_to_box2d(x_center, y_center, norm_w, norm_h, img_width, img_height)

                # Create the object dictionary in the new format
                object_dict = {
                    "category": CLASS_MAPPING[class_no],
                    "id": line_idx,
                    "attributes": {
                        "occluded": False,
                        "truncated": False,
                        "trafficLightColor": "none"
                    },
                    "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                }
                objects_list.append(object_dict)

        # If there are no valid objects, don't create a JSON file
        if not objects_list:
            continue

        # Assemble the final JSON structure for the image
        final_json_structure = {
            "name": base_name,  # The name is the file name without extension
            "attributes": {
                "weather": "undefined",
                "scene": "undefined",
                "timeofday": "undefined"
            },
            "frames": [
                {
                    "timestamp": 10000,  # Default timestamp
                    "objects": objects_list
                }
            ]
        }

        # Write the final dictionary to its own JSON file
        output_json_path = os.path.join(OUTPUT_DIR, base_name + ".json")
        with open(output_json_path, 'w') as f:
            json.dump(final_json_structure, f, indent=4)  # Use indent=4 to match example

    print(f"\nConversion complete! ✨ Check the '{OUTPUT_DIR}' directory.")


if __name__ == '__main__':
    convert_yolo_to_bdd_frames()