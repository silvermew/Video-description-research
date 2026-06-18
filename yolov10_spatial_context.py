import math
from ultralytics import YOLO

class YOLOSpatialContext:
    """
    A standalone module to run YOLOv10 object detection and calculate the 
    spatial relationship between detected objects and a human reference point.
    """
    
    def __init__(self, model_path="yolov10n.pt", conf_threshold=0.5, near_threshold_px=200):
        """
        Initializes the YOLO model.
        
        Args:
            model_path (str): Path to the YOLOv10 weights (e.g., 'yolov10n.pt').
                              Ultralytics will auto-download standard models.
            conf_threshold (float): Minimum confidence threshold for object detection.
            near_threshold_px (int): Pixel distance threshold to classify an object as 'near' vs 'far'.
        """
        print(f"Loading YOLO model from: {model_path}...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.near_threshold_px = near_threshold_px

    def _calculate_relative_position(self, obj_center_x, obj_center_y, ref_x, ref_y):
        """
        Calculates a human-readable relative position string.
        
        Args:
            obj_center_x (float): Object center X pixel coordinate.
            obj_center_y (float): Object center Y pixel coordinate.
            ref_x (float): Reference X pixel coordinate (e.g., human hip).
            ref_y (float): Reference Y pixel coordinate (e.g., human hip).
            
        Returns:
            str: e.g. 'near top-left', 'far bottom-right'
        """
        # Calculate Euclidean distance
        distance = math.hypot(obj_center_x - ref_x, obj_center_y - ref_y)
        
        # Determine proximity
        proximity = "near" if distance <= self.near_threshold_px else "far"
        
        # Determine Vertical position (Y goes down in images)
        if obj_center_y < ref_y - 30: # 30px buffer to prevent noise
            vertical = "above"
        elif obj_center_y > ref_y + 30:
            vertical = "below"
        else:
            vertical = "level with"
            
        # Determine Horizontal position
        if obj_center_x < ref_x - 30:
            horizontal = "left of"
        elif obj_center_x > ref_x + 30:
            horizontal = "right of"
        else:
            horizontal = "center of"
            
        # Clean up wording for direct matches
        if vertical == "level with" and horizontal == "center of":
            return f"{proximity} exactly at"
            
        return f"{proximity} {vertical} and {horizontal}"

    def process_frame(self, frame, reference_point):
        """
        Runs inference on the frame, tracks objects, and computes relative positions.
        
        Args:
            frame (np.ndarray): The RGB or BGR image frame from cv2.
            reference_point (tuple): A tuple (x, y) representing the pixel 
                                     coordinates of the human reference point 
                                     (e.g., from MediaPipe landmarks).
                                     
        Returns:
            list[dict]: A list of dictionaries containing detected objects and 
                        their relative positions to the human.
        """
        if reference_point is None or len(reference_point) != 2:
            return [] # No valid human reference point provided

        ref_x, ref_y = reference_point
        results = []
        
        # Run YOLO inference with tracking
        # persist=True ensures IDs are remembered across frames
        inference_results = self.model.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False)
        
        for result in inference_results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                conf = box.conf.item()
                if conf < self.conf_threshold:
                    continue
                    
                # Class name
                class_id = int(box.cls.item())
                base_class_name = self.model.names[class_id]
                
                # Tracking ID Extraction & Semantic Naming
                if box.id is not None:
                    track_id = int(box.id.item())
                    semantic_name = f"{base_class_name}_id_{track_id}"
                else:
                    semantic_name = base_class_name
                
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Calculate Object Center
                obj_center_x = (x1 + x2) / 2
                obj_center_y = (y1 + y2) / 2
                
                # Determine spatial relationship
                spatial_relationship = self._calculate_relative_position(
                    obj_center_x, obj_center_y, ref_x, ref_y
                )
                
                # Exclude objects that are physically 'far' from the human reference point
                if spatial_relationship.startswith("far"):
                    continue
                
                results.append({
                    'object': semantic_name,
                    'confidence': conf,
                    'position': f"{spatial_relationship} human",
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
                
        return results

# ==========================================
# EXAMPLE USAGE 
# ==========================================
if __name__ == "__main__":
    import cv2
    import numpy as np
    
    # 1. Initialize detector
    detector = YOLOSpatialContext(model_path="yolov10n.pt", conf_threshold=0.5)
    
    # 2. Create a dummy image
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 3. Define a fake human reference point (e.g., center of screen)
    # Note: If extracting from MediaPipe (which returns normalized 0.0 to 1.0 coords),
    # make sure to multiply by frame width and height!
    # e.g. ref_x = int(landmark.x * frame_width)
    human_hip_pixel_coords = (320, 240) 
    
    # 4. Process frame
    detected_context = detector.process_frame(dummy_frame, human_hip_pixel_coords)
    print("Detected Spatial Context:")
    for item in detected_context:
        print(item)
