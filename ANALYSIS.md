# Agent-S Issue #175: Coordinate Offset Problem - Analysis

## Problem Summary
Users report that Agent-S has poor clicking accuracy, especially for small UI elements (buttons, icons, checkboxes). The grounding model produces coordinates that are slightly off-target.

## Current Coordinate Flow

```
1. generate_coords() 
   → Grounding model (UI-TARS) receives screenshot + element description
   → Returns raw coordinates as text (e.g., "(523, 847)")
   → Regex extracts integers: re.findall(r"\d+", response)

2. resize_coordinates()
   → Scales from grounding model dimensions to screen dimensions
   → x_new = x * screen_width / grounding_width
   → y_new = y * screen_height / grounding_height
   → Uses round() for final values
```

## Root Causes Identified

### 1. Single-Point Prediction Variance
The grounding model outputs a single (x, y) point. For the same query, different calls can produce slightly different coordinates due to:
- Model stochasticity (temperature > 0)
- Ambiguous element boundaries
- No confidence score to filter low-quality predictions

### 2. Resolution Mismatch Amplification
When `grounding_width != screen_width`, small errors get amplified:
- Example: grounding at 1000px, screen at 1920px → 1.92x amplification
- A 5px error at grounding resolution becomes ~10px error on screen

### 3. No Element Size Awareness
The system doesn't know how big the target element is:
- For a 200px button, 10px offset = still hits
- For a 15px icon, 10px offset = complete miss

### 4. Integer Rounding Accumulation
Using `round()` introduces ±0.5px error per dimension, cumulative with scaling.

## Proposed Solutions

### Solution A: Multi-Sample Voting (Low Effort, High Impact)
Run grounding model N times (N=3 or 5), take median of coordinates.
```python
def generate_coords_robust(self, ref_expr: str, obs: Dict, n_samples: int = 3) -> List[int]:
    samples = []
    for _ in range(n_samples):
        self.grounding_model.reset()
        prompt = f"Query:{ref_expr}\nOutput only the coordinate of one point.\n"
        self.grounding_model.add_message(
            text_content=prompt, image_content=obs["screenshot"], put_text_last=True
        )
        response = call_llm_safe(self.grounding_model)
        coords = self._parse_coords(response)
        if coords:
            samples.append(coords)
    
    if not samples:
        raise ValueError("Failed to get valid coordinates")
    
    # Median is more robust to outliers than mean
    x = int(np.median([s[0] for s in samples]))
    y = int(np.median([s[1] for s in samples]))
    return [x, y]
```

**Pros:** Simple, effective for variance reduction
**Cons:** 3x latency increase (can parallelize), 3x API cost

### Solution B: Bounding Box + Center (Medium Effort)
Modify prompt to request bounding box instead of point:
```python
prompt = f"Query:{ref_expr}\nOutput the bounding box [x1, y1, x2, y2] of the element.\n"
# Then use center: ((x1+x2)/2, (y1+y2)/2)
```

**Pros:** More stable than point prediction, gives element size info
**Cons:** May need prompt engineering per grounding model

### Solution C: Hybrid OCR Fallback (Medium Effort)
For text-containing elements, use OCR as ground truth when available:
```python
def generate_coords(self, ref_expr: str, obs: Dict) -> List[int]:
    # Try OCR first for text elements
    ocr_coords = self._try_ocr_match(ref_expr, obs)
    if ocr_coords and self._is_high_confidence(ocr_coords):
        return ocr_coords
    
    # Fall back to grounding model
    return self._generate_coords_from_model(ref_expr, obs)
```

**Pros:** OCR gives pixel-perfect text locations
**Cons:** Only works for text elements, not icons

### Solution D: Hierarchical Refinement (High Effort, Highest Accuracy)
Two-stage approach:
1. First pass: Get rough coordinates
2. Crop 200x200px region around coords
3. Second pass: Re-run grounding on cropped region (higher effective resolution)
4. Map refined coords back to original image

```python
def generate_coords_refined(self, ref_expr: str, obs: Dict) -> List[int]:
    # Stage 1: Coarse localization
    coarse_coords = self._generate_coords_from_model(ref_expr, obs)
    
    # Stage 2: Crop and refine
    crop_size = 200
    x, y = coarse_coords
    crop = self._crop_region(obs["screenshot"], x, y, crop_size)
    
    # Re-run grounding on crop
    fine_coords = self._generate_coords_from_model(ref_expr, {"screenshot": crop})
    
    # Map back: fine coords are relative to crop center
    final_x = x + (fine_coords[0] - crop_size // 2)
    final_y = y + (fine_coords[1] - crop_size // 2)
    return [final_x, final_y]
```

**Pros:** Best accuracy, especially for dense UIs
**Cons:** 2x latency, more complex implementation

## Recommended Approach

Start with **Solution A (Multi-Sample Voting)** as it's:
- Quick to implement (~30 lines of code)
- Backward compatible (add as optional parameter)
- Measurably effective (can A/B test on OSWorld benchmark)

Then layer **Solution C (OCR Fallback)** since pytesseract is already integrated.

## Files to Modify
- `gui_agents/s3/agents/grounding.py`: Main changes in `generate_coords()` and `resize_coordinates()`
- Optionally add config flag in CLI args

## Testing Plan
1. Create a test with a UI containing various button sizes (10px, 20px, 40px, 80px icons)
2. Measure click success rate before/after changes
3. Compare latency impact

## Related
- PR #159: Handles normalized coordinates (different problem, complementary fix)
- Issue #168: Image limit errors (unrelated)
