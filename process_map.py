import cv2
import numpy as np
import json
import os
import math

def process_map(image_filename, output_filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_image_path = os.path.join(script_dir, image_filename)
    full_output_path = os.path.join(script_dir, output_filename)
    debug_image_path = os.path.join(script_dir, "debug_final_v18.jpg")

    print(f"Processing: {full_image_path}")
    img = cv2.imread(full_image_path)
    if img is None:
        print("ERROR: Could not load image.")
        return

    h_img, w_img, _ = img.shape
    debug_view = img.copy()

    # 1. Pre-processing
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 2. Color Ranges
    lower_white = np.array([0, 0, 215])
    upper_white = np.array([180, 40, 255])

    lower_yellow = np.array([15, 120, 120])
    upper_yellow = np.array([40, 255, 255])

    # Widen Red Range (V17 Fix)
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 80, 80])
    upper_red2 = np.array([180, 255, 255])

    # 3. Create Masks
    mask_struct_raw = cv2.inRange(hsv, lower_white, upper_white)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # 4. Cleaning & Fusing
    kernel_clean = np.ones((3,3), np.uint8)
    kernel_fuse = np.ones((5,5), np.uint8)
    
    mask_struct = cv2.morphologyEx(mask_struct_raw, cv2.MORPH_OPEN, kernel_clean)
    
    # === FIX: PRESERVE THIN RED FLOORS ===
    # Do not fuse red mask aggressively. Use raw or very light close.
    mask_red_fused = mask_red.copy() 
    
    mask_yellow_fused = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel_fuse)

    # ================= FIX 1: ROBUST HULL (CLEAN + GENERATE) =================
    # Building = Where there is ANY structure or floor.
    mask_building_mass = cv2.bitwise_or(mask_struct, mask_red_fused)
    mask_building_mass = cv2.bitwise_or(mask_building_mass, mask_yellow_fused)
    
    # === FIX: CLEAN NOISE BEFORE HULL ===
    # Open to remove small outside blobs that confuse the hull
    kernel_noise_remove = np.ones((5,5), np.uint8)
    mask_building_mass = cv2.morphologyEx(mask_building_mass, cv2.MORPH_OPEN, kernel_noise_remove)

    # Close to satisfy the "Solid Block" requirement
    # Reverted to 25x25 (Middle ground between 20 and 35)
    kernel_hull = np.ones((25,25), np.uint8) 
    mask_hull_generation = cv2.morphologyEx(mask_building_mass, cv2.MORPH_CLOSE, kernel_hull)
    
    # Dilate Hull for ROI checks
    kernel_spacer = np.ones((15,15), np.uint8)
    mask_hull_roi = cv2.dilate(mask_hull_generation, kernel_spacer, iterations=1)

    contours_hull_gen, _ = cv2.findContours(mask_hull_generation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_hull_roi, _ = cv2.findContours(mask_hull_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask_roi = np.zeros((h_img, w_img), dtype=np.uint8)
    hull_contour_gen = None
    
    if contours_hull_roi:
        roi_contour = max(contours_hull_roi, key=cv2.contourArea)
        cv2.drawContours(mask_roi, [roi_contour], -1, 255, -1)
        # cv2.drawContours(debug_view, [roi_contour], -1, (255, 0, 255), 2) # Dilated Hull (Optional)

    if contours_hull_gen:
        hull_contour_gen = max(contours_hull_gen, key=cv2.contourArea)
        cv2.drawContours(debug_view, [hull_contour_gen], -1, (255, 0, 255), 2) # Tight Hull

    # ================= FIX 2: WINDOWS (PERIMETER INTERSECTION) =================
    # Create a mask of the "Border Region"
    mask_perimeter = np.zeros((h_img, w_img), dtype=np.uint8)
    if hull_contour_gen is not None:
        cv2.drawContours(mask_perimeter, [hull_contour_gen], -1, 255, 30) # Draw thick border (30px)
        
    # Intersect White Struct with Perimeter
    mask_edge_structures = cv2.bitwise_and(mask_struct, mask_perimeter)
    
    contours_edge, _ = cv2.findContours(mask_edge_structures, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_window_contours = []
    
    for cnt in contours_edge:
        area = cv2.contourArea(cnt)
        if area < 30: continue
        if area > 1000: continue # Windows aren't huge
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        
        solidity = float(area) / hull_area
        
        x,y,w,h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h if h>0 else 0
        if aspect_ratio < 1.0: aspect_ratio = 1.0/aspect_ratio if aspect_ratio>0 else 0
        
        # Windows Check:
        # 1. Low Solidity (Spindly/Linear) -> < 0.7
        # 2. High Aspect Ratio -> > 2.0
        
        is_window = False
        if solidity < 0.65:
            is_window = True
        elif aspect_ratio > 1.8:
            is_window = True
            
        if is_window:
            valid_window_contours.append(cnt)
            cv2.rectangle(debug_view, (x,y), (x+w, y+h), (255, 255, 0), 2) # Cyan

    # ================= PATTERN ANALYSIS =================
    def analyze_pattern(object_mask, area, hsv_image):
        hue, sat, val = cv2.split(hsv_image)
        val = cv2.equalizeHist(val) 
        edges = cv2.Canny(val, 50, 150)
        internal_edges = cv2.bitwise_and(edges, edges, mask=object_mask)
        lines = cv2.HoughLinesP(internal_edges, 1, np.pi/180, threshold=15, minLineLength=10, maxLineGap=10)

        if lines is None:
            return {"density": 0, "coherence": 0, "lines": [], "is_solid": True}
        
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if angle < 0: angle += 180
            angles.append(angle)

        hist, bins = np.histogram(angles, bins=18, range=(0, 180))
        max_lines_in_bucket = np.max(hist)
        total_lines = len(angles)
        
        coherence = max_lines_in_bucket / total_lines if total_lines > 0 else 0
        density = (total_lines * 1000) / area if area > 0 else 0
        mean_s, std_s = cv2.meanStdDev(sat, mask=object_mask)
        is_solid = std_s[0][0] < 20 
        
        return {"density": density, "coherence": coherence, "lines": lines, "is_solid": is_solid}

    map_objects = []

    def contours_to_json(contours, default_type, obj_category):
        processed_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50: continue
            
            # --- ROI CHECK (Use DILATED Mask) ---
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if mask_roi[cY, cX] == 0:
                    continue 

            epsilon = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            points = [[int(p[0][0] - w_img/2), int(-(p[0][1] - h_img/2))] for p in approx]

            obj_type = default_type
            single_obj_mask = np.zeros((h_img, w_img), dtype="uint8")
            cv2.drawContours(single_obj_mask, [cnt], -1, 255, -1)

            if obj_category == "floor":
                stats = analyze_pattern(single_obj_mask, area, hsv)
                if obj_type == "floor_los": 
                    if stats["lines"] is None or len(stats["lines"]) < 2 or stats["coherence"] < 0.3:
                        continue 
                    for line in stats["lines"]:
                        cv2.line(debug_view, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 2)
                elif obj_type == "floor_trap":
                    if stats["density"] < 0.3: continue
                    for line in stats["lines"]:
                        cv2.line(debug_view, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 255, 0), 2)

            if obj_category == "wall":
                yellow_overlap = cv2.countNonZero(cv2.bitwise_and(single_obj_mask, mask_yellow_fused))
                overlap_ratio_y = yellow_overlap / area 
                red_overlap = cv2.countNonZero(cv2.bitwise_and(single_obj_mask, mask_red_fused))
                overlap_ratio_r = red_overlap / area
                
                if overlap_ratio_y > 0.3: 
                    stats = analyze_pattern(single_obj_mask, area, hsv)
                    if stats["density"] > 0.4: 
                        obj_type = "wall_breakable"
                        for line in stats["lines"]:
                            cv2.line(debug_view, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 0, 0), 2)
                elif overlap_ratio_r > 0.3:
                     obj_type = "wall_los"

            map_objects.append({"category": obj_category, "type": obj_type, "points": points})
            processed_count += 1
        return processed_count

    # Process Windows
    windows_to_export = contours_to_json(valid_window_contours, "window", "window")

    # Walls
    # Subtract Windows from Struct
    mask_windows_found = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.drawContours(mask_windows_found, valid_window_contours, -1, 255, -1)
    
    mask_walls_clean = cv2.subtract(mask_struct, mask_windows_found)
    contours_struct, _ = cv2.findContours(mask_walls_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    walls_found = contours_to_json(contours_struct, "wall_solid", "wall")

    # Floors
    mask_floors_area = cv2.subtract(mask_roi, mask_struct) 
    
    floors_yellow = cv2.bitwise_and(mask_yellow_fused, mask_yellow_fused, mask=mask_floors_area)
    floors_yellow = cv2.morphologyEx(floors_yellow, cv2.MORPH_OPEN, kernel_clean)
    cnt_y, _ = cv2.findContours(floors_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    floors_y_found = contours_to_json(cnt_y, "floor_trap", "floor")
    
    # Red Floors (Unfiltered for thin lines)
    floors_red = cv2.bitwise_and(mask_red_fused, mask_red_fused, mask=mask_floors_area)
    cnt_r, _ = cv2.findContours(floors_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    floors_r_found = contours_to_json(cnt_r, "floor_los", "floor")

    output_data = {"dimensions": {"width": w_img, "height": h_img}, "objects": map_objects}

    cv2.imwrite(debug_image_path, debug_view)
    print(f"V18 Debug saved to: {debug_image_path}")

    with open(full_output_path, 'w') as f:
        json.dump(output_data, f)
        print(f"V18 Done. Windows: {windows_to_export}, Walls: {walls_found}, Floors: {floors_y_found + floors_r_found}")

if __name__ == "__main__":
    process_map('map.jpg', 'map_data.json')