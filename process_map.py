import cv2
import numpy as np
import json
import os

def process_map(image_filename, output_filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_image_path = os.path.join(script_dir, image_filename)
    full_output_path = os.path.join(script_dir, output_filename)

    print(f"Processing: {full_image_path}")
    img = cv2.imread(full_image_path)
    if img is None:
        print("ERROR: Could not load image.")
        return

    h, w, _ = img.shape

    # =================NEW IN V2: COLOR SEGMENTATION=================
    # Convert to HSV color space for better color isolation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- Define HSV Color Ranges (Tweaked for R6 map colors) ---
    # White (Structure) - High Value, Low Saturation
    lower_white = np.array([0, 0, 210])
    upper_white = np.array([180, 50, 255])

    # Yellow/Orange (Breakable)
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    # Red (Line of Sight) - Red wraps around 0 in HSV, so we need two ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # --- Create Masks ---
    mask_struct = cv2.inRange(hsv, lower_white, upper_white)
    
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Clean up masks (remove noise)
    kernel_clean = np.ones((3,3), np.uint8)
    mask_struct = cv2.morphologyEx(mask_struct, cv2.MORPH_OPEN, kernel_clean)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel_clean)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel_clean)
    # ===============================================================

    map_objects = []

    # --- Helper Function to process contours ---
    def contours_to_json(contours, default_type, obj_category):
        processed_count = 0
        for cnt in contours:
            # Lowered threshold to catch smaller pillars/windows
            if cv2.contourArea(cnt) < 30: 
                continue
            
            epsilon = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            points = []
            for point in approx:
                x_coord, y_coord = point[0]
                points.append([int(x_coord - w/2), int(y_coord - h/2)])

            obj_type = default_type

            # =================NEW IN V2: CLASSIFICATION=================
            # Create a mask for just this current object
            single_obj_mask = np.zeros(mask_struct.shape, dtype="uint8")
            cv2.drawContours(single_obj_mask, [cnt], -1, 255, -1)

            # Check if this object overlaps significantly with Red or Yellow masks
            # We use bitwise_and to find the intersection area
            yellow_overlap = cv2.countNonZero(cv2.bitwise_and(single_obj_mask, mask_yellow))
            red_overlap = cv2.countNonZero(cv2.bitwise_and(single_obj_mask, mask_red))
            
            overlap_threshold = cv2.contourArea(cnt) * 0.2 # If 20% of the object is colored

            if obj_category == "wall":
                 # If it's a wall, color determines its type
                if yellow_overlap > overlap_threshold:
                    obj_type = "wall_breakable"
                elif red_overlap > overlap_threshold:
                    obj_type = "wall_los"
            # ===========================================================

            map_objects.append({
                "category": obj_category, # "wall" or "floor"
                "type": obj_type, 
                "points": points
            })
            processed_count += 1
        return processed_count


    # 1. Process Structural Walls (White)
    contours_struct, _ = cv2.findContours(mask_struct, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours_struct)} structural candidates.")
    walls_found = contours_to_json(contours_struct, "wall_solid", "wall")

    # 2. Process Special Floors
    # We need to find red/yellow areas that AREN'T walls.
    # Subtract structural mask from color masks.
    floors_yellow_mask = cv2.subtract(mask_yellow, mask_struct)
    floors_red_mask = cv2.subtract(mask_red, mask_struct)

    # Clean them up again after subtraction
    floors_yellow_mask = cv2.morphologyEx(floors_yellow_mask, cv2.MORPH_OPEN, kernel_clean)
    floors_red_mask = cv2.morphologyEx(floors_red_mask, cv2.MORPH_OPEN, kernel_clean)

    contours_floor_yellow, _ = cv2.findContours(floors_yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_floor_red, _ = cv2.findContours(floors_red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    floors_y_found = contours_to_json(contours_floor_yellow, "floor_trap", "floor")
    floors_r_found = contours_to_json(contours_floor_red, "floor_los", "floor")

    # Output Data Structure
    output_data = {
        "dimensions": {"width": w, "height": h},
        "objects": map_objects
    }

    with open(full_output_path, 'w') as f:
        json.dump(output_data, f)
        print(f"V2 Finished. Extracted {walls_found} walls, {floors_y_found + floors_r_found} special floors.")

    # Optional Debug: Save the masks to see what the computer sees
    # cv2.imwrite('debug_mask_white.jpg', mask_struct)
    # cv2.imwrite('debug_mask_yellow.jpg', mask_yellow)
    # cv2.imwrite('debug_mask_red.jpg', mask_red)

if __name__ == "__main__":
    process_map('map.jpg', 'map_data.json')