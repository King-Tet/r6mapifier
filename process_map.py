import cv2
import numpy as np
import json
import os
import math

def process_map(image_filename, output_filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_image_path = os.path.join(script_dir, image_filename)
    full_output_path = os.path.join(script_dir, output_filename)
    debug_image_path = os.path.join(script_dir, "debug_final_v16.jpg")

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
    # 2. Color Ranges
    lower_white = np.array([0, 0, 215])
    upper_white = np.array([180, 40, 255])

    lower_yellow = np.array([15, 120, 120])
    upper_yellow = np.array([40, 255, 255])

    # (STRICT for Hull/Windows, LOOSE for Floors)
    # Strict (Old) - Keeps hull tight, preserves windows
    lower_red1_strict = np.array([0, 100, 100])
    upper_red1_strict = np.array([10, 255, 255])
    lower_red2_strict = np.array([160, 100, 100])
    upper_red2_strict = np.array([180, 255, 255])

    # Loose (New) - Catch dark floors
    lower_red1_loose = np.array([0, 60, 60])
    upper_red1_loose = np.array([10, 255, 255])
    lower_red2_loose = np.array([160, 60, 60])
    upper_red2_loose = np.array([180, 255, 255])

    # 3. Create Masks
    mask_struct_raw = cv2.inRange(hsv, lower_white, upper_white)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Strict Red Masks
    mask_red1_strict = cv2.inRange(hsv, lower_red1_strict, upper_red1_strict)
    mask_red2_strict = cv2.inRange(hsv, lower_red2_strict, upper_red2_strict)
    mask_red_strict = cv2.bitwise_or(mask_red1_strict, mask_red2_strict)

    # Loose Red Masks
    mask_red1_loose = cv2.inRange(hsv, lower_red1_loose, upper_red1_loose)
    mask_red2_loose = cv2.inRange(hsv, lower_red2_loose, upper_red2_loose)
    mask_red_loose = cv2.bitwise_or(mask_red1_loose, mask_red2_loose)

    # 4. Cleaning & Fusing
    kernel_clean = np.ones((3,3), np.uint8)
    kernel_fuse = np.ones((5,5), np.uint8)
    
    mask_struct = cv2.morphologyEx(mask_struct_raw, cv2.MORPH_OPEN, kernel_clean)
    
    # Fused Masks
    mask_red_strict_fused = cv2.morphologyEx(mask_red_strict, cv2.MORPH_CLOSE, kernel_fuse)
    
    # SPLIT LOGIC (V10): Use OPEN instead of CLOSE for loose red. 
    # carpets are often connected to floors by thin pixel bridges. 
    # Open will break these bridges.
    mask_red_loose_fused = cv2.morphologyEx(mask_red_loose, cv2.MORPH_OPEN, kernel_fuse)
    
    mask_yellow_fused = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel_fuse)

    # ================= HULL CALCULATIONS =================
    # Building = Where there is ANY structure or floor.
    # USE STRICT MASK for Hull to avoid closing windows
    mask_building_mass = cv2.bitwise_or(mask_struct, mask_red_strict_fused)
    mask_building_mass = cv2.bitwise_or(mask_building_mass, mask_yellow_fused)
    
    # 1. Tight Hull (For Window Detection)
    kernel_hull = np.ones((20,20), np.uint8) 
    mask_hull_closed = cv2.morphologyEx(mask_building_mass, cv2.MORPH_CLOSE, kernel_hull)
    
    contours_tight, _ = cv2.findContours(mask_hull_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tight_hull_contour = max(contours_tight, key=cv2.contourArea) if contours_tight else None

    # 2. Dilated Hull (For Output Spacer & Object Filtering)
    # User wanted a Spacer to avoid cutting off windows.
    kernel_spacer = np.ones((11,11), np.uint8)
    mask_hull_dilated = cv2.dilate(mask_hull_closed, kernel_spacer, iterations=1)
    
    contours_dilated, _ = cv2.findContours(mask_hull_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask_roi = np.zeros((h_img, w_img), dtype=np.uint8)
    dilated_hull_contour = None
    
    if contours_dilated:
        dilated_hull_contour = max(contours_dilated, key=cv2.contourArea)
        cv2.drawContours(mask_roi, [dilated_hull_contour], -1, 255, -1)
        cv2.drawContours(debug_view, [dilated_hull_contour], -1, (255, 0, 255), 2) # Purple Hull (Spaced)

    # ================= WINDOWS (GAPS IN TIGHT MASS) =================
    # Use TIGHT hull for gaps.
    mask_gaps_raw = cv2.subtract(mask_hull_closed, mask_struct)
    mask_gaps = cv2.morphologyEx(mask_gaps_raw, cv2.MORPH_OPEN, kernel_clean)
    
    contours_gaps, _ = cv2.findContours(mask_gaps, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_window_contours = []
    
    if tight_hull_contour is not None:
        for cnt in contours_gaps:
            area = cv2.contourArea(cnt)
            if area < 80: continue 
            if area > 3000: continue 
            
            x,y,w,h = cv2.boundingRect(cnt)
            
            # Aspect Ratio Filter (Ignore Squares)
            aspect_ratio = float(w)/h if h > 0 else 0
            if aspect_ratio < 1: aspect_ratio = 1.0 / aspect_ratio
            
            # Windows are usually thin (High Aspect Ratio)
            if aspect_ratio < 1.3: # Relaxed slightly from 1.5 
                continue 

            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Check proximity to TIGHT Hull Edge
            dist = cv2.pointPolygonTest(tight_hull_contour, (cX, cY), True)
            
            if 0 <= dist < 35: 
                valid_window_contours.append(cnt)
                cv2.rectangle(debug_view, (x,y), (x+w, y+h), (255, 255, 0), 2) # Cyan

    # ================= PATTERN ANALYSIS =================
    def analyze_pattern(object_mask, area, hsv_image):
        hue, sat, val = cv2.split(hsv_image)
        val = cv2.equalizeHist(val) 
        edges = cv2.Canny(val, 50, 150)
        
        # Erode mask to avoid picking up boundary edges as lines
        # This fixes the "Curved Hallway" issue where outline lines > diagonal lines
        kernel_erode = np.ones((3,3), np.uint8)
        mask_internal = cv2.erode(object_mask, kernel_erode, iterations=1)
        
        internal_edges = cv2.bitwise_and(edges, edges, mask=mask_internal)
        lines = cv2.HoughLinesP(internal_edges, 1, np.pi/180, threshold=15, minLineLength=10, maxLineGap=10)

        if lines is None:
            return {"density": 0, "coherence": 0, "lines": [], "is_solid": True, "diag_ratio": 0, "std": 0}
        
        angles = []
        diag_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if angle < 0: angle += 180
            angles.append(angle)
            
            # Check diagonal (approx 45 or 135 deg, +/- 15 deg tolerance)
            # 45 deg +/- 15 -> 30-60
            # 135 deg +/- 15 -> 120-150
            if (30 < angle < 60) or (120 < angle < 150):
                diag_lines += 1

        hist, bins = np.histogram(angles, bins=18, range=(0, 180))
        max_lines_in_bucket = np.max(hist)
        total_lines = len(angles)
        
        coherence = max_lines_in_bucket / total_lines if total_lines > 0 else 0
        density = (total_lines * 1000) / area if area > 0 else 0
        mean_s, std_s = cv2.meanStdDev(sat, mask=object_mask)
        is_solid = std_s[0][0] < 20 
        diag_ratio = diag_lines / total_lines if total_lines > 0 else 0
        
        return {"density": density, "coherence": coherence, "lines": lines, "is_solid": is_solid, "diag_ratio": diag_ratio, "std": std_s[0][0]}

    map_objects = []

    def contours_to_json(contours, default_type, obj_category):
        processed_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50: continue
            
            # --- ROI CHECK (Use DILATED Hull) ---
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
                    # Require diagonal lines logic to differentiate from carpets/solid red
                    
                    if stats["lines"] is None or len(stats["lines"]) < 2:
                        continue 

                    # AREA-BASED FILTER (V8):
                    # Assumption: Carpets/Noise are small (< 1000 px). Valid rooms/hallways are large (> 1000 px).
                    
                    diag = stats["diag_ratio"]
                    std = stats["std"]
                    
                    if area > 1000:
                         # Large Object (Hallway/Room): Relaxed checks
                         # V11 TUNING: Raised from 0.05 to 0.10 based on split stats.
                         # Carpet = 0.08 (REJECT)
                         # Hallway = 0.21 (ACCEPT)
                         if diag < 0.10: continue 
                         if std < 10: continue
                    else:
                         # Small Object (Likely Carpet/Artifact): Strict checks
                         if diag < 0.20: continue 
                         if std < 25: continue
                    print(f"ACCEPTED Floor: Area={area}, Diag={stats['diag_ratio']:.2f}, Std={stats['std']:.2f}, Coh={stats['coherence']:.2f}")

                    # Draw Stats on Image for Debugging
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # Draw Text with black border for visibility
                        label = f"D:{diag:.2f} C:{stats['coherence']:.2f}"
                        cv2.putText(debug_view, label, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                        cv2.putText(debug_view, label, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    # Draw OUTLINE (Cyan) to show the full detected area
                    cv2.drawContours(debug_view, [cnt], -1, (255, 255, 0), 3)

                    # Hide internal green lines (confusing user)
                    # if stats["lines"] is not None:
                    #     for line in stats["lines"]:
                    #         cv2.line(debug_view, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 2)
                elif obj_type == "floor_trap":
                    if stats["density"] < 0.3: continue
                    for line in stats["lines"]:
                        cv2.line(debug_view, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 255, 0), 2)

            if obj_category == "wall":
                yellow_overlap = cv2.countNonZero(cv2.bitwise_and(single_obj_mask, mask_yellow_fused))
                overlap_ratio_y = yellow_overlap / area 
                # Use LOOSE red for walls too, just in case
                red_overlap = cv2.countNonZero(cv2.bitwise_and(single_obj_mask, mask_red_loose_fused))
                overlap_ratio_r = red_overlap / area
                
                if overlap_ratio_y > 0.3: 
                    stats = analyze_pattern(single_obj_mask, area, hsv)
                    if stats["density"] > 0.4: 
                        obj_type = "wall_breakable"
                        for line in stats["lines"]:
                            cv2.line(debug_view, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 0, 0), 2)
                elif overlap_ratio_r > 0.3:
                     obj_type = "wall_los"
                     # Draw debug for wall_los so user can see what's being detected
                     cv2.drawContours(debug_view, [cnt], -1, (0, 0, 255), 2)

            map_objects.append({"category": obj_category, "type": obj_type, "points": points})
            processed_count += 1
        return processed_count

    # Process Windows
    windows_to_export = contours_to_json(valid_window_contours, "window", "window")

    # Walls
    contours_struct, _ = cv2.findContours(mask_struct, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    walls_found = contours_to_json(contours_struct, "wall_solid", "wall")

    # Floors
    # ================= FIX 3: PRESERVE THIN FLOORS =================
    mask_floors_area = cv2.subtract(mask_roi, mask_struct) 
    
    floors_yellow = cv2.bitwise_and(mask_yellow_fused, mask_yellow_fused, mask=mask_floors_area)
    cnt_y, _ = cv2.findContours(floors_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    floors_y_found = contours_to_json(cnt_y, "floor_trap", "floor")
    
    # Red Floors - Skip/Min cleaning
    # Use LOOSE red for floors
    floors_red = cv2.bitwise_and(mask_red_loose_fused, mask_red_loose_fused, mask=mask_floors_area)
    # No opening here to save thin lines
    cnt_r, _ = cv2.findContours(floors_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    floors_r_found = contours_to_json(cnt_r, "floor_los", "floor")

    output_data = {"dimensions": {"width": w_img, "height": h_img}, "objects": map_objects}

    cv2.imwrite(debug_image_path, debug_view)
    print(f"V16 Debug saved to: {debug_image_path}")

    with open(full_output_path, 'w') as f:
        json.dump(output_data, f)
        print(f"V16 Done. Windows: {windows_to_export}, Walls: {walls_found}, Floors: {floors_y_found + floors_r_found}")

if __name__ == "__main__":
    process_map('map.jpg', 'map_data.json')