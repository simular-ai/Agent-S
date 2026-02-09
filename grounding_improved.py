"""
Improved coordinate generation for Agent-S
Addresses Issue #175: Coordinate offset problem for small buttons

This file contains the improved generate_coords implementation.
To apply: replace the generate_coords method in gui_agents/s3/agents/grounding.py
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add this import at the top of grounding.py:
# import numpy as np
# from concurrent.futures import ThreadPoolExecutor, as_completed


def _parse_coordinates(response: str) -> Optional[Tuple[float, float]]:
    """
    Parse coordinates from grounding model response.
    Handles both integer (e.g., "523, 847") and normalized (e.g., "0.45, 0.72") formats.
    
    Returns None if parsing fails.
    """
    # Try to find floating point or integer coordinates
    # Pattern matches: (523, 847), [0.45, 0.72], 523 847, etc.
    numericals = re.findall(r"\d+\.?\d*", response)
    
    if len(numericals) < 2:
        return None
    
    x, y = float(numericals[0]), float(numericals[1])
    return (x, y)


def _single_grounding_call(grounding_model, ref_expr: str, screenshot, call_llm_safe) -> Optional[Tuple[float, float]]:
    """
    Make a single grounding model call and return parsed coordinates.
    """
    grounding_model.reset()
    prompt = f"Query:{ref_expr}\nOutput only the coordinate of one point in your response.\n"
    grounding_model.add_message(
        text_content=prompt, image_content=screenshot, put_text_last=True
    )
    
    response = call_llm_safe(grounding_model)
    return _parse_coordinates(response)


# ============================================================================
# REPLACE the existing generate_coords method with this improved version:
# ============================================================================

def generate_coords(self, ref_expr: str, obs: Dict, n_samples: int = 1) -> List[int]:
    """
    Generate coordinates for an element using the grounding model.
    
    Args:
        ref_expr: Natural language description of the element to locate
        obs: Dictionary containing 'screenshot' key with the screen image
        n_samples: Number of samples for multi-sample voting (default=1 for backward compat)
                   Set to 3 or 5 for improved accuracy on small elements
    
    Returns:
        [x, y] coordinates in grounding model resolution
    
    Notes:
        - When n_samples > 1, takes median of multiple predictions for robustness
        - Median is more robust to outliers than mean
    """
    if n_samples <= 1:
        # Original single-sample behavior for backward compatibility
        self.grounding_model.reset()
        prompt = f"Query:{ref_expr}\nOutput only the coordinate of one point in your response.\n"
        self.grounding_model.add_message(
            text_content=prompt, image_content=obs["screenshot"], put_text_last=True
        )
        
        response = call_llm_safe(self.grounding_model)
        print("RAW GROUNDING MODEL RESPONSE:", response)
        
        coords = _parse_coordinates(response)
        if coords is None:
            raise ValueError(f"Failed to parse coordinates from response: {response}")
        
        x, y = coords
        # Handle normalized coordinates (0-1 range)
        grounding_width = self.engine_params_for_grounding.get("grounding_width", 1920)
        grounding_height = self.engine_params_for_grounding.get("grounding_height", 1080)
        
        if x <= 1.0 and y <= 1.0:
            x = x * grounding_width
            y = y * grounding_height
        
        return [int(round(x)), int(round(y))]
    
    # Multi-sample voting for improved accuracy
    samples = []
    
    for i in range(n_samples):
        self.grounding_model.reset()
        prompt = f"Query:{ref_expr}\nOutput only the coordinate of one point in your response.\n"
        self.grounding_model.add_message(
            text_content=prompt, image_content=obs["screenshot"], put_text_last=True
        )
        
        response = call_llm_safe(self.grounding_model)
        print(f"GROUNDING SAMPLE {i+1}/{n_samples}: {response}")
        
        coords = _parse_coordinates(response)
        if coords is not None:
            x, y = coords
            
            # Handle normalized coordinates
            grounding_width = self.engine_params_for_grounding.get("grounding_width", 1920)
            grounding_height = self.engine_params_for_grounding.get("grounding_height", 1080)
            
            if x <= 1.0 and y <= 1.0:
                x = x * grounding_width
                y = y * grounding_height
            
            samples.append((x, y))
    
    if not samples:
        raise ValueError(f"Failed to get valid coordinates after {n_samples} attempts")
    
    # Use median for robustness to outliers
    x_coords = [s[0] for s in samples]
    y_coords = [s[1] for s in samples]
    
    final_x = int(round(np.median(x_coords)))
    final_y = int(round(np.median(y_coords)))
    
    print(f"MULTI-SAMPLE RESULT: samples={samples}, median=({final_x}, {final_y})")
    
    return [final_x, final_y]


# ============================================================================
# Alternative: OCR-Enhanced Grounding (for text elements)
# ============================================================================

def generate_coords_with_ocr_fallback(
    self, 
    ref_expr: str, 
    obs: Dict, 
    use_ocr_for_text: bool = True
) -> List[int]:
    """
    Generate coordinates with optional OCR fallback for text elements.
    
    If the reference expression appears to describe text and OCR finds a match,
    use the OCR bounding box center (more precise for text).
    Otherwise, fall back to the grounding model.
    """
    if use_ocr_for_text:
        # Try OCR first for text elements
        try:
            ocr_table, ocr_elements = self.get_ocr_elements(obs["screenshot"])
            
            # Simple text matching - look for ref_expr words in OCR results
            ref_words = set(ref_expr.lower().split())
            
            for elem in ocr_elements:
                elem_text = elem["text"].lower()
                if elem_text in ref_words or any(w in elem_text for w in ref_words if len(w) > 3):
                    # Found a matching text element - use its center
                    x = elem["left"] + elem["width"] // 2
                    y = elem["top"] + elem["height"] // 2
                    print(f"OCR MATCH: '{elem['text']}' at ({x}, {y})")
                    return [x, y]
        except Exception as e:
            print(f"OCR fallback failed: {e}")
    
    # Fall back to grounding model
    return self.generate_coords(ref_expr, obs)


# ============================================================================
# CLI argument addition (add to run.py and run_local.py argparse):
# ============================================================================
"""
parser.add_argument(
    "--grounding_samples",
    type=int,
    default=1,
    help="Number of grounding samples for multi-sample voting (1=disabled, 3-5 recommended for small elements)"
)
"""
