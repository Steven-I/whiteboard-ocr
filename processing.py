import cv2
import numpy as np

def normalize_resolution(image: np.ndarray, max_dim: int = 1920) -> tuple[np.ndarray, float]:
    """
    LO 1: Resizes input to standard dimensions to ensure scale-invariant 
    morphological and thresholding operations across different camera resolutions.
    """
    h, w = image.shape[:2]
    max_curr = max(h, w)
    if max_curr <= max_dim:
        return image, 1.0
    scale = max_dim / max_curr
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 quad points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    LO 1: Perspective Transformation to deskew tilted whiteboard camera angles 
    into a flat 2D plane.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    if max_width < 100 or max_height < 100:
        return image

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, transform_matrix, (max_width, max_height))

def isolate_whiteboard_roi(image: np.ndarray, enable_quad: bool = True) -> np.ndarray:
    """
    LO 1 & LO 2: Surface Isolation Pipeline.
    Combines Canny edge quad detection with HSV color-space masking to isolate 
    whiteboard panels and reject background monitors, walls, and ceiling fixtures.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_h, img_w = gray.shape[:2]

    # 1. Quad Contour Detection (Attempts 4-corner perspective warp)
    if enable_quad:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(contour)
                if area > (img_h * img_w * 0.20):
                    # Verify region isn't a highly saturated digital screen
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [approx], -1, 255, -1)
                    mean_sat = cv2.mean(hsv[:, :, 1], mask=mask)[0]
                    if mean_sat < 70:
                        return four_point_transform(image, approx.reshape(4, 2))

    # 2. HSV Bounding Fallback (Isolates high-value, low-saturation board area)
    board_mask = cv2.inRange(hsv, (0, 0, 90), (180, 65, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_CLOSE, kernel)
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_OPEN, kernel)

    mask_contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if mask_contours:
        largest = max(mask_contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > (img_h * img_w * 0.15):
            x, y, w, h = cv2.boundingRect(largest)
            pad_x, pad_y = int(w * 0.015), int(h * 0.015)
            return image[y+pad_y:y+h-pad_y, x+pad_x:x+w-pad_x]

    return image

def neutralize_glare_and_lighting(
    image: np.ndarray, kernel_scale: float = 0.03, block_size: int = 25, c_val: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """
    LO 1 & LO 2: Advanced Illumination and Smudge Filter.
    - Min-Channel Projection: Maximizes contrast for green/red/blue markers.
    - Morphological Division (MORPH_CLOSE): Flatlines bright glare hotspots.
    - CLAHE: Balances localized contrast.
    - Sobel Edge Masking: Filters soft dry-erase wiping smudges.
    - Connected Components: Strips micro-noise specks (<15px).
    """
    if len(image.shape) == 3:
        b, g, r = cv2.split(image)
        # Projection of min(R,G,B) forces non-black ink to pop against white surfaces
        enhanced_gray = cv2.min(cv2.min(r, g), b)
    else:
        enhanced_gray = image.copy()

    denoised = cv2.medianBlur(enhanced_gray, 3)
    h, w = denoised.shape

    # 1. Background Estimation via Morphological Closing
    k_size = max(15, int(max(h, w) * kernel_scale))
    if k_size % 2 == 0:
        k_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    background = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    background[background == 0] = 1
    normalized = cv2.divide(denoised, background, scale=255).astype(np.uint8)

    # 2. Local Contrast Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    # 3. Adaptive Thresholding
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c_val
    )

    # 4. Sobel Gradient Magnitude Masking (Filters soft eraser smudges)
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobelx, sobely)
    grad_mask = (grad_mag > 20).astype(np.uint8) * 255
    cleaned_binary = cv2.bitwise_and(binary, grad_mask)

    # 5. Connected Component Area Filtering (Removes sensor noise specks)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_binary, connectivity=8)
    filtered_binary = np.zeros_like(cleaned_binary)
    min_area = max(12, int((h * w) * 0.00002))

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_binary[labels == i] = 255

    # Stroke reconnection
    stroke_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    filtered_binary = cv2.morphologyEx(filtered_binary, cv2.MORPH_CLOSE, stroke_kernel)

    return enhanced, filtered_binary

def cluster_boxes_into_lines(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """
    LO 3: Line Clustering Algorithm.
    Groups detected bounding boxes into natural reading order (top-to-bottom, left-to-right).
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[1])
    median_h = np.median([b[3] for b in boxes]) if boxes else 20.0

    lines = []
    current_line = [boxes[0]]

    for box in boxes[1:]:
        prev_box = current_line[-1]
        if abs(box[1] - prev_box[1]) < (median_h * 0.65):
            current_line.append(box)
        else:
            current_line = sorted(current_line, key=lambda b: b[0])
            lines.extend(current_line)
            current_line = [box]

    if current_line:
        current_line = sorted(current_line, key=lambda b: b[0])
        lines.extend(current_line)

    return lines

def extract_text_regions(binary_image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    LO 3: Bounding Box Proposal Extraction via Morphological Dilation and Contour Analysis.
    """
    img_h, img_w = binary_image.shape
    kw = max(10, int(img_w * 0.012))
    kh = max(3, int(img_h * 0.003))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    dilated = cv2.dilate(binary_image, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if w >= 10 and h >= 8 and area > 40 and area < (img_w * img_h * 0.65):
            boxes.append((x, y, w, h))

    return cluster_boxes_into_lines(boxes)

def create_highlight_overlay(
    image: np.ndarray, binary_mask: np.ndarray, boxes: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, float]:
    """
    Generates translucent green stroke highlighting strictly over text bounding boxes,
    and calculates board writing coverage percentage.
    """
    overlay = image.copy()
    clean_stroke_mask = np.zeros_like(binary_mask)

    for (x, y, w, h) in boxes:
        clean_stroke_mask[y:y+h, x:x+w] = binary_mask[y:y+h, x:x+w]

    green_layer = np.zeros_like(image, dtype=np.uint8)
    green_layer[clean_stroke_mask > 0] = [0, 255, 0]

    highlighted = cv2.addWeighted(overlay, 1.0, green_layer, 0.55, 0)

    for (x, y, w, h) in boxes:
        cv2.rectangle(highlighted, (x, y), (x + w, y + h), (0, 255, 0), 2)

    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
    stroke_pixels = np.count_nonzero(clean_stroke_mask)
    coverage_pct = (stroke_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    return highlighted, coverage_pct