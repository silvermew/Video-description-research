class InteractionDetector:
    """
    Calculates if a human's hand (wrists) is actively touching or intersecting 
    with a detected YOLO object by projecting YOLO Pose coordinates into pixel space.
    """
    def __init__(self, padding_pixels=20):
        """
        Args:
            padding_pixels (int): How much to expand the YOLO bounding box to create a 'touch zone'.
        """
        self.padding_pixels = padding_pixels
        # YOLO Pose (COCO format) wrist indices
        self.left_wrist_idx = 9
        self.right_wrist_idx = 10

    def detect_interactions(self, yolo_data, pose_keypoints_xyn, pose_keypoints_conf, frame_width, frame_height):
        """
        Checks for intersections between padded object bounding boxes and human hands.
        
        Args:
            yolo_data (list): List of dicts containing 'object' name and 'bbox' (x1, y1, x2, y2).
            pose_keypoints_xyn: YOLO Pose normalized keypoints tensor for the person [17, 2].
            pose_keypoints_conf: YOLO Pose confidence scores tensor for the person [17].
            frame_width (int): Pixel width of the video frame.
            frame_height (int): Pixel height of the video frame.
            
        Returns:
            list[str]: Highly explicit interaction flags.
        """
        flags = []
        if pose_keypoints_xyn is None or pose_keypoints_conf is None or not yolo_data:
            return ["[NO INTERACTION]"]

        # 1. Coordinate Mapping: Normalized (0.0 - 1.0) -> Pixel Space
        l_wrist_xyn = pose_keypoints_xyn[self.left_wrist_idx]
        r_wrist_xyn = pose_keypoints_xyn[self.right_wrist_idx]
        
        # Only use landmarks if they are reasonably visible (threshold > 0.3)
        l_visible = pose_keypoints_conf[self.left_wrist_idx] > 0.3
        r_visible = pose_keypoints_conf[self.right_wrist_idx] > 0.3
        
        l_x = int(l_wrist_xyn[0].item() * frame_width)
        l_y = int(l_wrist_xyn[1].item() * frame_height)
        
        r_x = int(r_wrist_xyn[0].item() * frame_width)
        r_y = int(r_wrist_xyn[1].item() * frame_height)

        interaction_found = False
        
        # 2. Intersection Math & Bounding Box Padding
        for obj in yolo_data:
            if 'bbox' not in obj:
                continue
                
            x1, y1, x2, y2 = obj['bbox']
            
            # Apply padding to create the 'touch zone'
            px1 = x1 - self.padding_pixels
            py1 = y1 - self.padding_pixels
            px2 = x2 + self.padding_pixels
            py2 = y2 + self.padding_pixels
            
            obj_name = obj['object']

            # Check left wrist intersection
            if l_visible and (px1 <= l_x <= px2 and py1 <= l_y <= py2):
                flags.append(f"[INTERACTION DETECTED: Left Hand intersecting {obj_name}]")
                interaction_found = True
            
            # Check right wrist intersection
            if r_visible and (px1 <= r_x <= px2 and py1 <= r_y <= py2):
                flags.append(f"[INTERACTION DETECTED: Right Hand intersecting {obj_name}]")
                interaction_found = True

        # 3. Interaction Flags
        if not interaction_found:
            return ["[NO INTERACTION]"]
            
        return flags
