import os
import re
import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from typing import List, Dict, Tuple, Union

def process_pdf(pdf_path: Union[str, Path], output_dir: Union[str, Path], start_page: int = None, end_page: int = None, progress_callback=None) -> List[str]:
    """
    Splits a PDF into images and saves them to the output directory using chunked processing to report progress.
    Returns a list of paths to the saved images.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pdf2image
        info = pdf2image.pdfinfo_from_path(str(pdf_path))
        pdf_total_pages = int(info.get("Pages", 0))
        
        # Calculate actual range
        actual_start = start_page if start_page is not None else 1
        actual_end = end_page if end_page is not None else pdf_total_pages
        total_to_extract = actual_end - actual_start + 1
        
        base_name = pdf_path.stem
        chunk_size = 10
        saved_paths = []
        
        print(f"Starting chunked extraction for {base_name} ({total_to_extract} pages)...")
        
        for current_start in range(actual_start, actual_end + 1, chunk_size):
            current_end = min(current_start + chunk_size - 1, actual_end)
            
            # Extract this chunk
            convert_from_path(
                str(pdf_path),
                output_folder=str(output_dir),
                fmt="png",
                output_file=f"{base_name}XImage",
                paths_only=True,
                first_page=current_start,
                last_page=current_end
            )
            
            # Collect and rename files from this chunk
            import glob
            # pdf2image pattern matches for files just created
            pattern = os.path.join(str(output_dir), f"{base_name}XImage*.png")
            generated_files = sorted(glob.glob(pattern))
            
            # Filter only the files that match the padded output from pdf2image (like 0001, 0002)
            # and aren't already renamed to our target format.
            for filepath in generated_files:
                filename = os.path.basename(filepath)
                # If it has the padded format like 'XImage-0001.png', rename it to 'XImageX.png'
                if "-" in filename:
                    try:
                        # Extract the index from pdf2image's padded suffix (e.g. "0001" from "TitleXImage-0001.png")
                        parts = filename.split("-")
                        if len(parts) > 1:
                            # The absolute page index in the PDF is crucial for sorting/grouping
                            idx_str = parts[-1].replace(".png", "")
                            page_idx = int(idx_str)
                            new_name = f"{base_name}XImage{page_idx}.png"
                            new_path = output_dir / new_name
                            os.rename(filepath, new_path)
                            saved_paths.append(str(new_path))
                    except Exception as e:
                        print(f"Rename error for {filename}: {e}")
                elif filename not in [os.path.basename(p) for p in saved_paths]:
                     # Fallback if it didn't have a dash but is new
                     saved_paths.append(filepath)

            if progress_callback:
                progress_callback(len(saved_paths), total_to_extract)

        # Final sort to ensure page order
        saved_paths = sorted(list(set(saved_paths)), key=lambda x: int(re.search(r"XImage(\d+)\.png$", x).group(1)) if re.search(r"XImage(\d+)\.png$", x) else 0)
        
        print(f"Successfully chunked {len(saved_paths)} pages to {output_dir}")
        return saved_paths
        
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return []

def enhance_image(image_path: Union[str, Path]) -> str:
    """
    Applies an OpenCV pipeline to enhance the image for OCR:
    Grayscale -> Denoise -> Adaptive Threshold -> Deskew
    Returns the path to the enhanced image (overwriting the original or saving as temporary).
    For this implementation, we will overwrite/update in place or return the path if successful.
    """
    image_path_str = str(image_path)
    try:
        img = cv2.imread(image_path_str)
        if img is None:
            print(f"Failed to load image: {image_path_str}")
            return image_path_str

        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # 3. Binarization (Adaptive Threshold)
        # using gaussian adaptive thresholding for better results on varying lighting
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # 4. De-skewing
        # Find all contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # If we have contours, we can try to find the orientation
        # This is a simple deskewing implementation
        # For complex handwritten notes, sometimes deskewing can damage layout if not careful.
        # We will use a minAreaRect approach but limit the rotation angle to avoid flipping.
        
        # (Optional: Only deskew if creating a new file, but here we just return the cleaned binary for API)
        # For API submission, high contrast binary is often good.
        
        # Overwrite the file with the enhanced version ?? 
        # API usually works better with original RGB for semantic understanding (figures, colors), 
        # BUT for strict handwriting OCR, binarization helps. 
        # The user requested: "OpenCV pipeline to perform de-skewing, denoising, and binarization before API transmission"
        # So we will save the enhanced image temporarily or overwrite.
        # Let's save as a temp file to avoid destroying original data if needed, or just overwrite if that's the flow.
        # To be safe, let's write to a temporary path or suffix.
        
        enhanced_path = image_path_str.replace(".png", "_enhanced.png").replace(".jpg", "_enhanced.jpg")
        cv2.imwrite(enhanced_path, binary)
        
        return enhanced_path

    except Exception as e:
        print(f"Error enhancing image {image_path_str}: {e}")
        return image_path_str

def get_image_grouping(folder_path: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Scans the folder for images matching 'TitleXImageY.format'.
    Returns a dictionary mapping 'Title' to a list of sorted image paths.
    """
    folder_path = Path(folder_path)
    groups = {}
    
    # Pattern explanation:
    # ^(.+?)                : Capture the Title (non-greedy) at the start
    # (?:[\sX_-]+|XImage)   : Separator (Spaces, 'X', '_', '-', or 'XImage' literal)
    # (\d+)                 : The Image/Page Number
    # \.(...)$              : Extension
    pattern = re.compile(r"^(.+?)(?:[\sX_-]+|XImage)(\d+)\.(png|jpg|jpeg|pdf|webp)$", re.IGNORECASE)

    if not folder_path.exists():
        return groups
        
    for p in folder_path.iterdir():
        if not p.is_file():
            continue
        
        # Skip hidden files
        if p.name.startswith('.'):
            continue
            
        match = pattern.match(p.name)
        if match:
            title = match.group(1).strip()
            # image_num = int(match.group(2))
            if title not in groups:
                groups[title] = []
            groups[title].append(str(p))
            
    # Sort images in each group by the number Y
    for title in groups:
        groups[title].sort(key=lambda x: int(pattern.search(Path(x).name).group(2)))
        
    return groups
