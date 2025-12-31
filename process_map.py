import cv2
import numpy as np
import json
import os

def process_map(image_filename, output_filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_image_path = os.path.join(script_dir, image_filename)
    full_output_path = os.path.join(script_dir, output_filename)
    debug_image_path = os.path.join(script_dir, "debug_lines.jpg")

    print(f"Processing: {full_image_path}")
    img = cv2.imread(full_image_path)
    if img is None:
        print("ERROR: Could not load image.")
        return

    h_img, w_img, _ = img.shape
    debug_view = img.copy()

    # 1. Pre-processing
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Color Ranges
    lower_white = np.array([0, 0, 210])
    upper_white = np.array([180, 50, 255])

    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # 3. Create Basic Masks
    mask_struct = cv2.inRange(hsv, lower_white, upper_white)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # 4. === KEY FIX 1: FUSING THE STRIPES ===
    # Use 'Morphological Close' to fill the black gaps between red stripes.
    # This turns the "patchy" floor into one solid blob.
    kernel_fuse = np.ones((5,5), np.uint8) 
    mask_red_fused = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel_fuse)
    # Also clean up noise
    mask_red_fused = cv2.morphologyEx(mask_red_fused, cv2.MORPH_OPEN, kernel_fuse)

    # Standard clean for others
    kernel_clean = np.ones((3,3), np.uint8)
    mask_struct = cv2.morphologyEx(mask_struct, cv2.MORPH_OPEN, kernel_clean)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel_clean)

    map_objects = []

    def contours_to_json(contours, default_type, obj_category):
        processed_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50: continue # Slight bump to ignore tiny noise
            
            epsilon = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            points = []
            for point in approx:
                x_coord, y_coord = point[0]
                points.append([int(x_coord - w_img/2), int(-(y_coord - h_img/2))])

            obj_type = default_type
            
            # Create a mask for just this object
            single_obj_mask = np.zeros(mask_struct.shape, dtype="uint8")
            cv2.drawContours(single_obj_mask, [cnt], -1, 255, -1)

            # Check Overlaps
            yellow_overlap = cv2.countNonZero(cv2.bitwise_and(single_obj_mask, mask_yellow))
            # Note: We check against the original 'mask_red', not the fused one, for overlap accuracy
            red_overlap = cv2.countNonZero(cv2.bitwise_and(single_obj_mask, mask_red))
            overlap_threshold = area * 0.2

            # === KEY FIX 2: HOUGH LINE DETECTION ===
            if obj_type == "floor_los":
                # A. ERODE the mask internally.
                # Why? We want to ignore the outer edges of the carpet.
                # A carpet has a square border (lines), but NO internal lines.
                kernel_erode = np.ones((3,3), np.uint8)
                internal_mask = cv2.erode(single_obj_mask, kernel_erode, iterations=2)
                
                # B. Get the Edges inside the eroded shape
                # Get the brightness
                hue, sat, val = cv2.split(hsv)
                # Boost contrast slightly to make stripes pop
                val = cv2.equalizeHist(val)
                # Canny Edge Detection
                edges = cv2.Canny(val, 50, 150)
                # Mask it so we only see edges INSIDE the shape
                internal_edges = cv2.bitwise_and(edges, edges, mask=internal_mask)

                # C. HOUGH LINES (The Carpet Killer)
                # minLineLength=20: Requires a straight line at least 20 pixels long.
                # Carpets have texture, but not 20px long straight lines.
                lines = cv2.HoughLinesP(internal_edges, 1, np.pi/180, threshold=20, minLineLength=25, maxLineGap=10)

                # D. DECISION
                if lines is None:
                    line_count = 0
                else:
                    line_count = len(lines)
                    # Debug: Draw the lines found in GREEN
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(debug_view, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Calculate Density (Lines per area)
                # Multiply by 1000 just to make the numbers readable (e.g., 0.5 instead of 0.0005)
                line_density = (line_count * 1000) / area
                
                print(f"Red Floor Check | Area: {area} | Lines Found: {line_count} | Density: {line_density:.2f}")

                # Threshold:
                # Carpet usually has 0 to 2 lines found (Density < 0.1)
                # Striped floor usually has 20+ lines (Density > 1.0)
                if line_density < 0.5:
                    print("-> REJECTED (Carpet - No internal lines)")
                    # Debug: Draw RED border on rejected carpet
                    cv2.drawContours(debug_view, [cnt], -1, (0, 0, 255), 2)
                    continue 
                else:
                    print("-> ACCEPTED (Striped Floor)")

            if obj_category == "wall":
                if yellow_overlap > overlap_threshold:
                    obj_type = "wall_breakable"
                elif red_overlap > overlap_threshold:
                    obj_type = "wall_los"

            map_objects.append({
                "category": obj_category,
                "type": obj_type, 
                "points": points
            })
            processed_count += 1
        return processed_count

    # Process Walls
    contours_struct, _ = cv2.findContours(mask_struct, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    walls_found = contours_to_json(contours_struct, "wall_solid", "wall")

    # Process Floors
    # Use the FUSED mask for floors to get big contiguous shapes
    floors_yellow_mask = cv2.subtract(mask_yellow, mask_struct)
    floors_red_mask = cv2.subtract(mask_red_fused, mask_struct)
    
    floors_yellow_mask = cv2.morphologyEx(floors_yellow_mask, cv2.MORPH_OPEN, kernel_clean)
    floors_red_mask = cv2.morphologyEx(floors_red_mask, cv2.MORPH_OPEN, kernel_clean)

    contours_floor_yellow, _ = cv2.findContours(floors_yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_floor_red, _ = cv2.findContours(floors_red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    floors_y_found = contours_to_json(contours_floor_yellow, "floor_trap", "floor")
    floors_r_found = contours_to_json(contours_floor_red, "floor_los", "floor")

    output_data = {
        "dimensions": {"width": w_img, "height": h_img},
        "objects": map_objects
    }

    cv2.imwrite(debug_image_path, debug_view)
    print(f"Debug saved to: {debug_image_path}")

    with open(full_output_path, 'w') as f:
        json.dump(output_data, f)
        print(f"V5 Done. Walls: {walls_found}, Floors: {floors_y_found + floors_r_found}")

if __name__ == "__main__":
    process_map('map.jpg', 'map_data.json')