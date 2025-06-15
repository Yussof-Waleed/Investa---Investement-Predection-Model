"""
National ID Card Processing Module

This module handles the processing, validation, and storage of Egyptian national ID cards.
It uses computer vision and OCR to extract data from ID card images.
"""

import os
import re
import cv2
import tempfile
import easyocr
from rest_framework.decorators import api_view
from django.http import JsonResponse
from ultralytics import YOLO

# ====================================================
# CONFIGURATION AND INITIALIZATION
# ====================================================

# Initialize EasyOCR reader (done once for efficiency)
reader = easyocr.Reader(['ar'], gpu=False)

# Helper function to get model paths relative to this file
def get_model_path(model_name):
    """Get the path to a model file in the saved_models directory."""
    return os.path.join(os.path.dirname(__file__), 'saved_models', model_name)

# ====================================================
# UTILITY FUNCTIONS
# ====================================================

def expand_bbox_height(bbox, scale=1.2, image_shape=None):
    """
    Expand the height of a bounding box while keeping the width constant.
    
    Args:
        bbox: List containing [x1, y1, x2, y2] coordinates
        scale: Factor by which to scale the height
        image_shape: Original image dimensions to prevent out-of-bounds coordinates
        
    Returns:
        List with the new bbox coordinates [x1, new_y1, x2, new_y2]
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width // 2
    center_y = y1 + height // 2
    new_height = int(height * scale)
    new_y1 = max(center_y - new_height // 2, 0)
    new_y2 = min(center_y + new_height // 2, image_shape[0]) if image_shape else center_y + new_height // 2
    return [x1, new_y1, x2, new_y2]

def preprocess_image(cropped_image):
    """Convert image to grayscale for better OCR results."""
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)   
    return gray_image

# ====================================================
# OCR AND TEXT EXTRACTION
# ====================================================

def extract_text(image, bbox, lang='ara'):
    """
    Extract text from a specified region in the image.
    
    Args:
        image: The source image
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        lang: Language for OCR ('ara' for Arabic, 'eng' for English)
        
    Returns:
        Extracted text as string
    """
    x1, y1, x2, y2 = bbox
    cropped_image = image[y1:y2, x1:x2]
    preprocessed_image = preprocess_image(cropped_image)
    results = reader.readtext(preprocessed_image, detail=0, paragraph=True)
    text = ' '.join(results)
    return text.strip()

def detect_national_id(cropped_image):
    """
    Detect and extract national ID digits from a cropped image.
    
    Args:
        cropped_image: Image containing only the ID number area
        
    Returns:
        String containing the detected ID number
    """
    model = YOLO(get_model_path('detect_id.pt'))
    results = model(cropped_image)
    detected_info = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_info.append((cls, x1))
            cv2.rectangle(cropped_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cropped_image, str(cls), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Sort by x-coordinate to get digits in correct order
    detected_info.sort(key=lambda x: x[1])
    id_number = ''.join([str(cls) for cls, _ in detected_info])
    
    return id_number

# ====================================================
# ID CARD PROCESSING
# ====================================================

def process_image(cropped_image):
    """
    Extract all relevant information from a cropped ID card image.
    
    Args:
        cropped_image: The cropped image containing just the ID card
        
    Returns:
        Tuple containing extracted information:
        (first_name, second_name, merged_name, nid, address, birth_date, governorate, gender)
    """
    # Load the trained YOLO model for objects (fields) detection
    model = YOLO(get_model_path('detect_odjects.pt'))
    results = model(cropped_image)

    # Variables to store extracted values
    first_name = ''
    second_name = ''
    merged_name = ''
    nid = ''
    address = ''
    serial = ''

    # Loop through the results
    for result in results:
        output_path = 'd2.jpg'
        result.save(output_path)

        for box in result.boxes:
            bbox = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]
            bbox = [int(coord) for coord in bbox]

            if class_name == 'firstName':
                first_name = extract_text(cropped_image, bbox, lang='ara')
            elif class_name == 'lastName':
                second_name = extract_text(cropped_image, bbox, lang='ara')
            elif class_name == 'serial':
                serial = extract_text(cropped_image, bbox, lang='eng')
            elif class_name == 'address':
                address = extract_text(cropped_image, bbox, lang='ara')
            elif class_name == 'nid':
                expanded_bbox = expand_bbox_height(bbox, scale=1.5, image_shape=cropped_image.shape)
                cropped_nid = cropped_image[expanded_bbox[1]:expanded_bbox[3], expanded_bbox[0]:expanded_bbox[2]]
                nid = detect_national_id(cropped_nid)

    merged_name = f"{first_name} {second_name}"
    print(f"First Name: {first_name}")
    print(f"Second Name: {second_name}")
    print(f"Full Name: {merged_name}")
    print(f"National ID: {nid}")
    print(f"Address: {address}")
    print(f"Serial: {serial}")

    # Decode additional information from the ID number
    decoded_info = decode_egyptian_id(nid)
    return (first_name, second_name, merged_name, nid, address, decoded_info["Birth Date"], 
            decoded_info["Governorate"], decoded_info["Gender"])

def detect_and_process_id_card(image_path):
    """
    Detect an ID card in an image and extract information from it.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple containing extracted ID information
    """
    # Load the ID card detection model
    id_card_model = YOLO(get_model_path('detect_id_card.pt'))
    
    # Perform inference to detect the ID card
    id_card_results = id_card_model(image_path)

    # Load the original image using OpenCV
    image = cv2.imread(image_path)

    # Crop the ID card from the image
    cropped_image = None
    for result in id_card_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            cropped_image = image[y1:y2, x1:x2]
            break  # Take the first detected ID card
        if cropped_image is not None:
            break
    
    if cropped_image is None:
        raise ValueError("No ID card detected in the image")

    # Pass the cropped image to the processing function
    return process_image(cropped_image)

def decode_egyptian_id(id_number):
    """
    Decode Egyptian national ID number to extract demographic information.
    
    Args:
        id_number: 14-digit Egyptian national ID number
        
    Returns:
        Dictionary with decoded information (birth date, governorate, gender)
    """
    if not id_number or len(id_number) != 14 or not id_number.isdigit():
        return {
            'Birth Date': 'Unknown',
            'Governorate': 'Unknown',
            'Gender': 'Unknown'
        }
        
    governorates = {
        '01': 'Cairo',
        '02': 'Alexandria',
        '03': 'Port Said',
        '04': 'Suez',
        '11': 'Damietta',
        '12': 'Dakahlia',
        '13': 'Ash Sharqia',
        '14': 'Kaliobeya',
        '15': 'Kafr El - Sheikh',
        '16': 'Gharbia',
        '17': 'Monoufia',
        '18': 'El Beheira',
        '19': 'Ismailia',
        '21': 'Giza',
        '22': 'Beni Suef',
        '23': 'Fayoum',
        '24': 'El Menia',
        '25': 'Assiut',
        '26': 'Sohag',
        '27': 'Qena',
        '28': 'Aswan',
        '29': 'Luxor',
        '31': 'Red Sea',
        '32': 'New Valley',
        '33': 'Matrouh',
        '34': 'North Sinai',
        '35': 'South Sinai',
        '88': 'Foreign'
    }

    try:
        century_digit = int(id_number[0])
        year = int(id_number[1:3])
        month = int(id_number[3:5])
        day = int(id_number[5:7])
        governorate_code = id_number[7:9]
        gender_code = int(id_number[12:13])

        if century_digit == 2:
            full_year = 1900 + year
        elif century_digit == 3:
            full_year = 2000 + year
        else:
            raise ValueError("Invalid century digit")
            
        # Validate date
        if month < 1 or month > 12 or day < 1 or day > 31:
            raise ValueError("Invalid date in ID")

        gender = "Male" if gender_code % 2 != 0 else "Female"
        governorate = governorates.get(governorate_code, "Unknown")
        birth_date = f"{full_year:04d}-{month:02d}-{day:02d}"

        return {
            'Birth Date': birth_date,
            'Governorate': governorate,
            'Gender': gender
        }
    except (ValueError, IndexError) as e:
        print(f"Error decoding ID number: {str(e)}")
        return {
            'Birth Date': 'Unknown',
            'Governorate': 'Unknown',
            'Gender': 'Unknown'
        }

# ====================================================
# API ENDPOINTS
# ====================================================

@api_view(['POST'])
def upload_national_card(request):
    """
    Upload and validate national ID card front and back images.
    
    Performs validation on:
    - Required fields
    - Image format and size
    - ID card detection
    - ID information extraction
    
    Args:
        request: HTTP request with form data containing:
            - id_front_image: Front image of the national ID card
            - id_back_image: Back image of the national ID card
            - user_id: User identifier
            
    Returns:
        JsonResponse with upload result or error details
    """
    errors = {}
    temp_files = []
    
    try:
        # Required field validation
        front_national_card = request.FILES.get('id_front_image')
        back_national_card = request.FILES.get('id_back_image')
        user_id = request.data.get('user_id')
        
        if not user_id:
            errors['user_id'] = 'User ID is required'
        if not front_national_card:
            errors['front_image'] = 'Front image is required'
        if not back_national_card:
            errors['back_image'] = 'Back image is required'
            
        if errors:
            return JsonResponse({'error': 'Missing required fields', 'details': errors}, status=400)
            
        # File type validation
        allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
        max_size = 5 * 1024 * 1024  # 5MB limit
        
        if front_national_card.content_type not in allowed_types:
            errors['front_image'] = f'Invalid format: {front_national_card.content_type}. Use JPEG/PNG only'
        elif front_national_card.size > max_size:
            errors['front_image'] = 'Image exceeds 5MB size limit'
            
        if back_national_card.content_type not in allowed_types:
            errors['back_image'] = f'Invalid format: {back_national_card.content_type}. Use JPEG/PNG only'
        elif back_national_card.size > max_size:
            errors['back_image'] = 'Image exceeds 5MB size limit'
            
        if errors:
            return JsonResponse({'error': 'Invalid files', 'details': errors}, status=400)
        
        # Save images to temporary files for processing
        front_temp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        for chunk in front_national_card.chunks():
            front_temp.write(chunk)
        front_temp.close()
        temp_files.append(front_temp.name)
        
        back_temp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        for chunk in back_national_card.chunks():
            back_temp.write(chunk)
        back_temp.close()
        temp_files.append(back_temp.name)
        
        # ID card validation for front image
        try:
            # Detect if image contains an ID card
            id_card_model = YOLO(get_model_path('detect_id_card.pt'))
            front_results = id_card_model(front_temp.name)
            
            if len(front_results[0].boxes) == 0:
                errors['front_image'] = 'No national ID card detected in the image'
            else:
                # Try to extract information from the ID card
                try:
                    id_info = detect_and_process_id_card(front_temp.name)
                    id_number = id_info[3]  # National ID number
                    
                    # Validate ID number (14 digits for Egyptian IDs)
                    if not id_number or not re.match(r'^\d{14}$', id_number):
                        errors['front_image'] = 'Invalid or unreadable ID number'
                except Exception as e:
                    errors['front_image'] = f'Cannot extract information from ID card: {str(e)[:100]}. Please provide a clearer image'
        except Exception as e:
            errors['front_image'] = f'Failed to validate ID card: {str(e)[:100]}'
            
        # Basic validation for back image
        try:
            back_results = id_card_model(back_temp.name)
            if len(back_results[0].boxes) == 0:
                errors['back_image'] = 'No national ID card detected in the back image'
        except Exception as e:
            errors['back_image'] = f'Failed to validate back of ID card: {str(e)[:100]}'
            
        if errors:
            return JsonResponse({'error': 'ID validation failed', 'details': errors}, status=400)
        
        # If all validation passes, proceed with upload to Google Drive
        main_folder_id = "1nIPlwpcUGkDK0hvCfU_TlrRJyt6EmSi5"
        user_folder_id = get_or_create_drive_folder(user_id, parent_folder_id=main_folder_id)

        front_file_name = front_national_card.name
        front_image_url = upload_image_to_drive(front_national_card, front_file_name, user_folder_id)

        back_file_name = back_national_card.name
        back_image_url = upload_image_to_drive(back_national_card, back_file_name, user_folder_id)

        # Store the data in Firebase with extracted information
        ref = db.reference("users").child(user_id).child("national_card")
        data = {
            "front_url": front_image_url,
            "back_url": back_image_url,
            "verified": True
        }
        
        # Add extracted information if available
        if 'id_info' in locals() and id_info:
            data.update({
                "id_number": id_info[3],
                "full_name": id_info[2],
                "birth_date": id_info[5],
                "gender": id_info[7],
                "governorate": id_info[6]
            })
            
        ref.set(data)

        # Return the original response structure to maintain frontend contract
        return JsonResponse({
            'message': 'National card uploaded.',
            'front_url': front_image_url,
            'back_url': back_image_url
        })

    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)[:200]}'}, status=500)
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass  # Suppress cleanup errors