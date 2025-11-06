import dlib
import cv2
import os
import numpy as np
import json
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
import time

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'model/shape_predictor_68_face_landmarks.dat')

# Normalize landmarks
def normalize_landmarks(landmarks, face_width, face_height):
    return [(x / face_width, y / face_height) for x, y in landmarks]

# Calculate Euclidean distance
def calculate_landmark_distance(landmarks1, landmarks2):
    return sum(euclidean((x1, y1), (x2, y2)) for (x1, y1), (x2, y2) in zip(landmarks1, landmarks2))

# Calculate additional metrics
def calculate_face_metrics(landmarks):
    # Calculate eye aspect ratio (simplified)
    left_eye_width = euclidean(landmarks[36], landmarks[39])
    left_eye_height = euclidean(landmarks[37], landmarks[41])
    right_eye_width = euclidean(landmarks[42], landmarks[45])
    right_eye_height = euclidean(landmarks[43], landmarks[47])
    
    eye_ratio = (left_eye_height/left_eye_width + right_eye_height/right_eye_width) / 2 if left_eye_width > 0 and right_eye_width > 0 else 0.3
    
    # Calculate mouth aspect ratio
    mouth_width = euclidean(landmarks[48], landmarks[54])
    mouth_height = euclidean(landmarks[51], landmarks[57])
    mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0.5
    
    return {
        'eye_aspect_ratio': round(eye_ratio, 4),
        'mouth_aspect_ratio': round(mouth_ratio, 4),
        'symmetry_score': round(calculate_symmetry_score(landmarks), 4)
    }

def calculate_symmetry_score(landmarks):
    # Calculate facial symmetry by comparing left and right side landmarks
    try:
        left_points = [landmarks[i] for i in range(0, 16)]  # Left face contour
        right_points = [landmarks[i] for i in range(16, 27)]  # Right face contour
        
        symmetry_errors = []
        for i in range(min(len(left_points), len(right_points))):
            # Mirror right points and compare with left
            mirrored_right = (1 - right_points[i][0], right_points[i][1])  # Simple mirror
            symmetry_errors.append(euclidean(left_points[i], mirrored_right))
        
        return np.mean(symmetry_errors) if symmetry_errors else 0.1
    except:
        return 0.1

# Main function to call
def run_deepfake_detection(folder_path, frame_ids):
    start_time = time.time()
    image_extensions = ('.jpg', '.jpeg', '.png')
    landmarks_list = []
    image_files = []
    face_metrics_list = []

    all_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)])
    
    total_faces_detected = 0
    analysis_details = []

    for file in all_files:
        path = os.path.join(folder_path, file)
        try:
            img = cv2.imread(path)
            if img is None:
                print(f"{file} - Error: Could not read image")
                analysis_details.append({
                    'filename': file,
                    'face_detected': False,
                    'landmarks_count': 0,
                    'metrics': None,
                    'status': 'error_reading'
                })
                continue

            # Convert to grayscale properly
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if not faces:
                print(f"{file} - Face: NO")
                analysis_details.append({
                    'filename': file,
                    'face_detected': False,
                    'landmarks_count': 0,
                    'metrics': None,
                    'status': 'no_face'
                })
                continue

            shape = predictor(gray, faces[0])
            coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
            face_width = faces[0].width()
            face_height = faces[0].height()
            normalized_coords = normalize_landmarks(coords, face_width, face_height)

            landmarks_list.append(normalized_coords)
            image_files.append(file)

            # Calculate face metrics
            metrics = calculate_face_metrics(normalized_coords)
            face_metrics_list.append(metrics)

            analysis_details.append({
                'filename': file,
                'face_detected': True,
                'landmarks_count': len(coords),
                'face_confidence': 0.95,  # Placeholder for face detection confidence
                'metrics': metrics,
                'status': 'analyzed'
            })

            print(f"{file} - Face: YES | Landmarks Detected: {len(coords)}")
            total_faces_detected += 1

        except Exception as e:
            print(f"{file} - Error: {str(e)}")
            analysis_details.append({
                'filename': file,
                'face_detected': False,
                'landmarks_count': 0,
                'metrics': None,
                'status': f'error: {str(e)}'
            })
            continue

    processing_time = time.time() - start_time

    if len(landmarks_list) < 2:
        return {
            "total_images": len(all_files),
            "genuine_count": 0,
            "deepfake_count": 0,
            "genuine_percentage": 0,
            "deepfake_percentage": 0,
            "video_classification": "Insufficient Data",
            "confidence_score": 0,
            "analysis_metrics": {
                "total_faces_detected": total_faces_detected,
                "processing_time": round(processing_time, 2),
                "frames_per_second": round(len(all_files) / processing_time, 2) if processing_time > 0 else 0,
                "average_landmarks_per_face": 0,
                "consistency_score": 0,
                "analysis_details": analysis_details
            },
            "risk_assessment": "LOW",
            "recommendation": "Upload video with more clear faces for accurate analysis"
        }, total_faces_detected, {}

    def compare_landmark_consistency(landmarks_list):
        consistency_scores = []
        for i in range(len(landmarks_list)):
            for j in range(i + 1, len(landmarks_list)):
                distance = calculate_landmark_distance(landmarks_list[i], landmarks_list[j])
                consistency_scores.append(distance)
                print(f"Comparing {image_files[i]} and {image_files[j]} - Distance: {distance:.2f}")
        return np.array(consistency_scores)

    consistency_scores = compare_landmark_consistency(landmarks_list)

    # Handle case where we don't have enough data for clustering
    if len(consistency_scores) < 2:
        return {
            "total_images": len(all_files),
            "genuine_count": len(landmarks_list),
            "deepfake_count": 0,
            "genuine_percentage": 100,
            "deepfake_percentage": 0,
            "video_classification": "genuine",
            "confidence_score": 0.7,
            "analysis_metrics": {
                "total_faces_detected": total_faces_detected,
                "processing_time": round(processing_time, 2),
                "frames_per_second": round(len(all_files) / processing_time, 2) if processing_time > 0 else 0,
                "average_landmarks_per_face": round(np.mean([detail['landmarks_count'] for detail in analysis_details if detail['face_detected']]), 2),
                "consistency_score": 0,
                "analysis_details": analysis_details
            },
            "risk_assessment": "LOW",
            "recommendation": "Limited data for analysis - more frames recommended"
        }, total_faces_detected, {}

    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(consistency_scores.reshape(-1, 1))
    labels = kmeans.labels_

    deepfake_label = 1 if np.mean(consistency_scores[labels == 1]) > np.mean(consistency_scores[labels == 0]) else 0

    genuine_count = 0
    deepfake_count = 0
    frame_predictions = []

    for i, file in enumerate(image_files):
        label = labels[i]
        status = "deepfake" if label == deepfake_label else "genuine"
        
        # Calculate confidence based on distance from cluster center
        distance_to_center = abs(consistency_scores[i] - kmeans.cluster_centers_[label][0])
        max_distance = np.max(np.abs(consistency_scores - kmeans.cluster_centers_[label][0]))
        confidence = max(0.6, 1 - (distance_to_center / max_distance)) if max_distance > 0 else 0.8

        if status == "genuine":
            genuine_count += 1
        else:
            deepfake_count += 1

        frame_predictions.append({
            'filename': file,
            'status': status,
            'confidence': round(confidence, 3),
            'cluster_label': int(label),
            'consistency_score': round(float(consistency_scores[i]), 4)
        })

        print(f"{file} - {status} (Confidence: {confidence:.2f})")

    total_images = len(all_files)
    genuine_percentage = (genuine_count / total_images) * 100
    deepfake_percentage = (deepfake_count / total_images) * 100

    # Calculate overall confidence
    overall_confidence = np.mean([pred['confidence'] for pred in frame_predictions]) if frame_predictions else 0.7
    
    # Risk assessment
    if deepfake_percentage > 70:
        risk_level = "HIGH"
        recommendation = "This video shows strong signs of being manipulated"
    elif deepfake_percentage > 40:
        risk_level = "MEDIUM"
        recommendation = "This video shows moderate signs of manipulation"
    else:
        risk_level = "LOW"
        recommendation = "This video appears to be genuine"

    print("\n--- Overall Classification ---")
    print(f"Total Images Processed: {total_images}")
    print(f"Genuine Images: {genuine_count} ({genuine_percentage:.2f}%)")
    print(f"Deepfake Images: {deepfake_count} ({deepfake_percentage:.2f}%)")
    print(f"Overall Confidence: {overall_confidence:.2f}")
    print(f"Risk Level: {risk_level}")

    video_classification = "genuine" if genuine_percentage > 50 else "deepfake"

    summary = {
        "total_images": total_images,
        "genuine_count": genuine_count,
        "deepfake_count": deepfake_count,
        "genuine_percentage": round(genuine_percentage, 2),
        "deepfake_percentage": round(deepfake_percentage, 2),
        "video_classification": video_classification,
        "confidence_score": round(overall_confidence, 3),
        "analysis_metrics": {
            "total_faces_detected": total_faces_detected,
            "processing_time": round(processing_time, 2),
            "frames_per_second": round(len(all_files) / processing_time, 2) if processing_time > 0 else 0,
            "average_landmarks_per_face": round(np.mean([detail['landmarks_count'] for detail in analysis_details if detail['face_detected']]), 2),
            "consistency_score": round(float(np.mean(consistency_scores)), 4),
            "analysis_details": analysis_details
        },
        "risk_assessment": risk_level,
        "recommendation": recommendation,
        "frame_predictions": frame_predictions
    }

    frame_statuses = dict(zip(image_files, ["genuine" if labels[i] != deepfake_label else "deepfake" for i in range(len(image_files))]))

    # Save frame statuses to results folder
    video_name = os.path.basename(folder_path)
    result_path = os.path.join('results', f'{video_name}_frame_status.json')
    with open(result_path, 'w') as f:
        json.dump(frame_statuses, f)

    # Save detailed analysis
    detailed_result_path = os.path.join('results', f'{video_name}_detailed_analysis.json')
    with open(detailed_result_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary, total_faces_detected, frame_statuses
