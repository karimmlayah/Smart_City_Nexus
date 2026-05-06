"""
License Plate OCR and Verification Module
Extracts text from plate images and verifies Tunisian plate format
"""
import os
import re
import base64
import requests
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import json


class PlateOCR:
    """Handle OCR extraction and verification of license plates"""
    
    def __init__(self, ocr_api_key: str, openai_api_key: str):
        self.ocr_api_key = ocr_api_key
        self.openai_api_key = openai_api_key
        self.ocr_url = "https://api.apilayer.com/image_to_text/upload"
    
    def extract_text_from_image(self, image_path: Path) -> Optional[str]:
        """
        Extract text from plate image using OCR API
        
        Args:
            image_path: Path to the plate image
            
        Returns:
            Extracted text or None if failed
        """
        try:
            print(f"\n📤 Sending to OCR API:")
            print(f"   Image: {image_path}")
            print(f"   Exists: {image_path.exists()}")
            
            if not image_path.exists():
                print(f"   ✗ Image file does not exist!")
                return None
            
            file_size = image_path.stat().st_size
            print(f"   Size: {file_size} bytes")
            
            # Check if image is too small (API requires minimum size)
            # API rejects images < ~2KB, so resize anything < 5KB to be safe
            if file_size < 5000:  # Less than 5KB
                print(f"   ⚠ Image small ({file_size} bytes), resizing for better OCR...")
                # Resize image to make it larger
                import cv2
                img = cv2.imread(str(image_path))
                if img is not None:
                    # Scale up by 5x for very small images
                    height, width = img.shape[:2]
                    scale_factor = 5
                    new_width = width * scale_factor
                    new_height = height * scale_factor
                    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                    
                    # Save to temp file with higher quality
                    temp_path = image_path.parent / f"temp_resized_{image_path.name}"
                    cv2.imwrite(str(temp_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    # Use resized image
                    image_path = temp_path
                    file_size = temp_path.stat().st_size
                    print(f"   ✓ Resized to: {new_width}x{new_height}, new size: {file_size} bytes")
                else:
                    print(f"   ✗ Could not read image with cv2")
            
            # Read image file
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            print(f"   Data size: {len(image_data)} bytes")
            
            headers = {
                "apikey": self.ocr_api_key
            }
            
            print(f"   URL: {self.ocr_url}")
            print(f"   API Key: {self.ocr_api_key[:10]}...")
            
            response = requests.post(
                self.ocr_url,
                headers=headers,
                data=image_data,
                timeout=10
            )
            
            print(f"\n📥 OCR API Response:")
            print(f"   Status: {response.status_code}")
            print(f"   Headers: {dict(response.headers)}")
            print(f"   Body (first 500 chars): {response.text[:500]}")
            
            if response.status_code == 200:
                # API returns JSON: {"all_text": "200-9271", "annotations": [...], "lang": "und"}
                try:
                    result = response.json()
                    print(f"   Parsed JSON: {result}")
                    text = result.get("all_text", "").strip()
                    if text:
                        print(f"   ✓ OCR extracted: {text}")
                        return text
                    else:
                        print(f"   ⚠ OCR returned empty text: {result}")
                        return None
                except json.JSONDecodeError as je:
                    print(f"   ⚠ JSON decode error: {je}")
                    # Fallback to plain text
                    text = response.text.strip()
                    print(f"   ✓ OCR extracted (plain): {text}")
                    return text if text else None
            else:
                print(f"   ✗ OCR API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"✗ OCR extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def verify_tunisian_plate_with_openai(self, ocr_text: str, image_path: Path) -> Dict[str, Any]:
        """
        Verify and correct text from image using OpenAI Vision
        
        Args:
            ocr_text: Raw OCR extracted text (e.g., "200-9271", "200.9271", "2009271")
            image_path: Path to plate image for visual verification
            
        Returns:
            Dict with verified plate info
        """
        try:
            # First, auto-format: replace common separators with تونس
            formatted_text = ocr_text
            separators = ['-', '_', '.', '/', '|', ':', ';', ',', '*', '+', '=', ' ']
            
            for sep in separators:
                if sep in formatted_text:
                    formatted_text = formatted_text.replace(sep, ' تونس ', 1)  # Replace first occurrence
                    break
            
            # If no separator found, try to insert تونس in the middle
            if 'تونس' not in formatted_text and 'TN' not in formatted_text:
                numbers = re.findall(r'\d+', formatted_text)
                if len(numbers) >= 2:
                    formatted_text = f"{numbers[0]} تونس {numbers[-1]}"
                elif len(numbers) == 1 and len(numbers[0]) >= 3:
                    # Single concatenated number
                    num = numbers[0]
                    mid = len(num) // 2
                    formatted_text = f"{num[:mid]} تونس {num[mid:]}"
            
            formatted_text = ' '.join(formatted_text.split())  # Clean extra spaces
            
            print(f"   Auto-formatted: {ocr_text} → {formatted_text}")
            
            # Encode image to base64
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Simple prompt - don't mention license plates
            prompt = f"""Look at this image and read the text carefully.

The OCR system extracted: "{ocr_text}"

Your task:
1. Look at the image and identify all visible text/numbers
2. Correct any OCR errors (like O vs 0, I vs 1, etc.)
3. Return the corrected text

Return ONLY a JSON object:
{{
    "corrected_text": "the corrected text you see",
    "confidence": 0.0-1.0,
    "notes": "what corrections you made, if any"
}}

Be precise with numbers. If you see the number 0, write 0 (not O)."""

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.1
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    verified_data = json.loads(json_match.group())
                    corrected = verified_data.get("corrected_text", ocr_text)
                    
                    # Apply our formatting: replace any separator with تونس
                    corrected_formatted = corrected
                    separators = ['-', '_', '.', '/', '|', ':', ';', ',', '*', '+', '=']
                    for sep in separators:
                        if sep in corrected_formatted:
                            corrected_formatted = corrected_formatted.replace(sep, ' تونس ', 1)
                            break
                    
                    # If no separator, try to insert تونس
                    if 'تونس' not in corrected_formatted and 'TN' not in corrected_formatted:
                        numbers = re.findall(r'\d+', corrected_formatted)
                        if len(numbers) >= 2:
                            corrected_formatted = f"{numbers[0]} تونس {numbers[-1]}"
                        elif len(numbers) == 1 and len(numbers[0]) >= 3:
                            num = numbers[0]
                            mid = len(num) // 2
                            corrected_formatted = f"{num[:mid]} تونس {num[mid:]}"
                    
                    corrected_formatted = ' '.join(corrected_formatted.split())
                    
                    # Extract numbers
                    numbers = re.findall(r'\d+', corrected_formatted)
                    left_numbers = numbers[0] if len(numbers) > 0 else ""
                    right_numbers = numbers[-1] if len(numbers) > 1 else (numbers[0] if len(numbers) == 1 else "")
                    
                    print(f"   OpenAI corrected: {corrected} → {corrected_formatted}")
                    
                    return {
                        "is_tunisian_plate": bool(numbers),
                        "plate_number": corrected_formatted,
                        "left_numbers": left_numbers,
                        "right_numbers": right_numbers,
                        "confidence": verified_data.get("confidence", 0.8),
                        "notes": verified_data.get("notes", "OpenAI verified")
                    }
                else:
                    print(f"   Could not parse OpenAI response: {content}")
                    return self._fallback_verification(ocr_text)
            else:
                print(f"   OpenAI API error: {response.status_code}")
                return self._fallback_verification(ocr_text)
                
        except Exception as e:
            print(f"   OpenAI verification failed: {e}")
            return self._fallback_verification(ocr_text)
    
    def _fallback_verification(self, ocr_text: str) -> Dict[str, Any]:
        """
        Fallback verification - automatically format Tunisian plates
        Handles ANY separator: -, _, ., /, |, :, ;, comma, *, +, =, space, or no separator
        """
        # Auto-format: replace separators with تونس
        formatted = ocr_text
        separators = ['-', '_', '.', '/', '|', ':', ';', ',', '*', '+', '=']
        
        for sep in separators:
            if sep in formatted:
                formatted = formatted.replace(sep, ' تونس ', 1)  # Replace first occurrence
                break
        
        # If already has تونس or TN, keep it
        if 'تونس' not in formatted and 'TN' not in formatted:
            # Try to find numbers and insert تونس
            numbers = re.findall(r'\d+', ocr_text)
            
            if len(numbers) >= 2:
                # Multiple number groups
                formatted = f"{numbers[0]} تونس {numbers[-1]}"
            elif len(numbers) == 1:
                # Single number - split it
                num = numbers[0]
                if len(num) >= 3:
                    # Try intelligent split
                    best_split = None
                    for split_pos in range(1, len(num)):
                        left = num[:split_pos]
                        right = num[split_pos:]
                        if 1 <= len(left) <= 4 and 1 <= len(right) <= 4:
                            best_split = (left, right)
                            break
                    
                    if best_split:
                        formatted = f"{best_split[0]} تونس {best_split[1]}"
                    else:
                        mid = len(num) // 2
                        formatted = f"{num[:mid]} تونس {num[mid:]}"
                else:
                    formatted = f"{num} تونس ?"
            else:
                formatted = ocr_text
        
        # Clean spaces
        formatted = ' '.join(formatted.split())
        
        # Extract numbers from formatted text
        numbers = re.findall(r'\d+', formatted)
        left_numbers = numbers[0] if len(numbers) > 0 else ""
        right_numbers = numbers[-1] if len(numbers) > 1 else ""
        
        return {
            "is_tunisian_plate": bool(numbers),
            "plate_number": formatted,
            "left_numbers": left_numbers,
            "right_numbers": right_numbers,
            "confidence": 0.7 if numbers else 0.3,
            "notes": f"Auto-formatted from '{ocr_text}'"
        }
    
    def process_plate_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Complete pipeline: OCR extraction + OpenAI verification
        
        Args:
            image_path: Path to plate image
            
        Returns:
            Dict with complete plate information
        """
        # Step 1: Extract text with OCR
        ocr_text = self.extract_text_from_image(image_path)
        
        if not ocr_text:
            return {
                "is_tunisian_plate": False,
                "plate_number": "OCR_FAILED",
                "left_numbers": "",
                "right_numbers": "",
                "confidence": 0.0,
                "notes": "OCR extraction failed",
                "ocr_raw": ""
            }
        
        # Step 2: Verify with OpenAI
        verified = self.verify_tunisian_plate_with_openai(ocr_text, image_path)
        verified["ocr_raw"] = ocr_text
        
        return verified


def process_violation_plates(
    car_id: int,
    plate_images: list,
    ocr_api_key: str,
    openai_api_key: str
) -> Dict[str, Any]:
    """
    Process all plate images for a car in violation
    Uses the highest confidence plate for OCR
    
    Args:
        car_id: Car tracking ID
        plate_images: List of (image_path, confidence) tuples
        ocr_api_key: OCR API key
        openai_api_key: OpenAI API key
        
    Returns:
        Dict with plate number and verification info
    """
    if not plate_images:
        return {
            "car_id": car_id,
            "plate_number": "NO_PLATE_DETECTED",
            "confidence": 0.0,
            "is_tunisian_plate": False
        }
    
    # Sort by confidence and get the best one
    sorted_plates = sorted(plate_images, key=lambda x: x[1], reverse=True)
    best_plate_path, best_confidence = sorted_plates[0]
    
    print(f"\n🔍 Processing plate for Car ID {car_id}")
    print(f"   Using best plate: {best_plate_path.name} (confidence: {best_confidence:.2%})")
    
    # Initialize OCR processor
    ocr = PlateOCR(ocr_api_key, openai_api_key)
    
    # Process the best plate image
    result = ocr.process_plate_image(best_plate_path)
    
    # Add metadata
    result["car_id"] = car_id
    result["detection_confidence"] = best_confidence
    result["total_plate_images"] = len(plate_images)
    result["best_plate_path"] = str(best_plate_path)
    
    return result
