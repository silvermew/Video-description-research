import numpy as np
from scipy.signal import savgol_filter, find_peaks

class KinematicSegmenter:
    """
    A physics-based Kinematic Windowing (Semantic Chunking) system.
    Analyzes the velocity of the human skeleton to find natural pauses in movement
    (local minima in velocity). These pauses serve as the boundaries for video chunks.
    """
    
    def __init__(self, 
                 hip_idx=0, 
                 right_wrist_idx=26, 
                 left_wrist_idx=29, 
                 min_chunk_length=30, 
                 max_chunk_length=150,
                 smoothing_window=15,
                 poly_order=3):
        """
        Initializes the Kinematic Segmenter.
        
        Args:
            hip_idx (int): Marker index for the Hip (Center of Mass).
            right_wrist_idx (int): Marker index for the Right Wrist (e.g., pRightUlnarStyloid).
            left_wrist_idx (int): Marker index for the Left Wrist (e.g., pLeftUlnarStyloid).
            min_chunk_length (int): Minimum frames for a chunk to prevent rapid micro-chunking.
            max_chunk_length (int): Maximum frames before forcing a split.
            smoothing_window (int): Window size for Savitzky-Golay filter (must be odd).
            poly_order (int): Polynomial order for Savitzky-Golay filter.
        """
        self.hip_idx = hip_idx
        self.right_wrist_idx = right_wrist_idx
        self.left_wrist_idx = left_wrist_idx
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.smoothing_window = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1
        self.poly_order = poly_order

    def segment(self, motion_data):
        """
        Extracts semantic chunk boundaries based on kinematic energy.
        
        Args:
            motion_data (np.ndarray): Shape (num_frames, 192) or (num_frames, 64, 3).
                                      Represents 64 3D markers over time.
        
        Returns:
            list of tuple: [(start_frame, end_frame), ...] representing semantic chunks.
        """
        num_frames = motion_data.shape[0]
        
        # 1. Feature Extraction
        # Ensure data is in shape (num_frames, 64, 3)
        if motion_data.ndim == 2 and motion_data.shape[1] == 192:
            data = motion_data.reshape(num_frames, 64, 3)
        elif motion_data.ndim == 3 and motion_data.shape[1] == 64 and motion_data.shape[2] == 3:
            data = motion_data
        else:
            raise ValueError(f"Invalid motion_data shape {motion_data.shape}. Expected (N, 192) or (N, 64, 3).")
            
        # Extract the key joints: (num_frames, 3) for each
        hip_trajectory = data[:, self.hip_idx, :]
        r_wrist_trajectory = data[:, self.right_wrist_idx, :]
        l_wrist_trajectory = data[:, self.left_wrist_idx, :]
        
        # 2. Velocity Calculation (Vectorized)
        # Calculate frame-to-frame difference (velocity). We prepend the first frame to maintain length.
        hip_vel = np.linalg.norm(np.diff(hip_trajectory, axis=0, prepend=hip_trajectory[0:1, :]), axis=1)
        r_wrist_vel = np.linalg.norm(np.diff(r_wrist_trajectory, axis=0, prepend=r_wrist_trajectory[0:1, :]), axis=1)
        l_wrist_vel = np.linalg.norm(np.diff(l_wrist_trajectory, axis=0, prepend=l_wrist_trajectory[0:1, :]), axis=1)
        
        # Calculate 'Kinematic Energy' by averaging the velocities of key joints
        kinematic_energy = (hip_vel + r_wrist_vel + l_wrist_vel) / 3.0
        
        # 3. Temporal Smoothing
        # Apply Savitzky-Golay filter to remove micro-jitters
        if num_frames > self.smoothing_window:
            smoothed_energy = savgol_filter(kinematic_energy, window_length=self.smoothing_window, polyorder=self.poly_order)
        else:
            smoothed_energy = kinematic_energy
            
        # Ensure non-negative energy after smoothing
        smoothed_energy = np.maximum(smoothed_energy, 0)
        
        # 4. Boundary Detection
        # We want to find local minima in velocity (pauses in movement).
        # We can find minima by finding peaks in the inverted signal.
        max_energy = np.max(smoothed_energy) if np.max(smoothed_energy) > 0 else 1.0
        inverted_energy = max_energy - smoothed_energy
        
        # Find peaks in the inverted signal (which correspond to local minima in energy)
        # We use a distance constraint to prevent peaks too close to each other
        peaks, _ = find_peaks(inverted_energy, distance=self.min_chunk_length)
        
        # Convert peaks to a list of boundary frames and ensure 0 and num_frames are included
        boundaries = [0] + peaks.tolist() + [num_frames]
        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))
        
        # 5. Failsafes (min and max chunk lengths)
        chunks = []
        start_idx = boundaries[0]
        
        # Iterate over potential boundaries to form valid chunks
        for i in range(1, len(boundaries)):
            end_idx = boundaries[i]
            
            # If the chunk is getting too long without a natural pause, force a split
            while (end_idx - start_idx) > self.max_chunk_length:
                # Force split at max_chunk_length
                forced_end = start_idx + self.max_chunk_length
                chunks.append((start_idx, forced_end))
                start_idx = forced_end
                
            # Now handle the remaining segment (or if it was already valid)
            if (end_idx - start_idx) >= self.min_chunk_length:
                chunks.append((start_idx, end_idx))
                start_idx = end_idx
            elif i == len(boundaries) - 1 and len(chunks) > 0:
                # If the very last segment is too short, just merge it with the previous chunk
                # Make sure the end index is the actual final frame
                prev_start, _ = chunks.pop()
                chunks.append((prev_start, end_idx))
        
        # If no valid chunks were formed (e.g., video too short), return one whole chunk
        if not chunks:
            chunks.append((0, num_frames))
            
        return chunks
