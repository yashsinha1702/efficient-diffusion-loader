import cv2
import numpy as np
from PIL import Image
import random

class LayoutAugmenter:
    def __init__(self):
        pass

    def augment(self, pil_image, max_objects=3):
        """
        Robustly augments layout. 
        Guarantees the object will be scaled < 1.0 to ensure movement is possible.
        """
        img = np.array(pil_image)
        
        # Handle shape (H, W, C)
        if len(img.shape) == 2:
            h, w = img.shape
            img = np.expand_dims(img, axis=2) # Make 3D
        else:
            h, w, channels = img.shape

        # Create blank canvas
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Decide count
        num_objects = random.randint(1, max_objects)
        
        objects_placed = 0
        
        # Retry loop to ensure at least one object is placed
        attempts = 0
        while objects_placed < num_objects and attempts < 10:
            attempts += 1
            
            # FORCE SCALE < 0.8 so it definitely fits and can move
            # We avoid 1.0+ scales to prevent 'out of bounds' errors
            scale = random.uniform(0.4, 0.75) 
            
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize
            resized = cv2.resize(img, (new_w, new_h))
            
            # Ensure resized is 3D
            if len(resized.shape) == 2:
                resized = np.expand_dims(resized, axis=2)

            # Random Position (Now guaranteed to be positive range)
            max_x = w - new_w
            max_y = h - new_h
            
            # Safe logic: If scaling failed, fallback to 0
            start_x = random.randint(0, max(0, max_x))
            start_y = random.randint(0, max(0, max_y))
            
            # Random Flip
            if random.random() > 0.5:
                resized = cv2.flip(resized, 1)
                if len(resized.shape) == 2: resized = np.expand_dims(resized, axis=2)

            # Paste
            # Get Region of Interest (ROI) on canvas
            roi = canvas[start_y:start_y+new_h, start_x:start_x+new_w]
            
            # Safety check for size match
            if roi.shape[:2] != resized.shape[:2]:
                continue # Skip this attempt if sizes don't match
                
            combined = cv2.max(roi, resized)
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = combined
            objects_placed += 1

        # Fallback: If loop failed completely (rare), return original
        if objects_placed == 0:
            return Image.fromarray(img)
            
        return Image.fromarray(canvas)