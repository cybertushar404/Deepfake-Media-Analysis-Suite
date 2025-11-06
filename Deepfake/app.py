from flask import Flask, render_template, request, jsonify, send_from_directory, redirect
import os
import cv2
from predict import run_deepfake_detection
import json

app = Flask(__name__)
app.secret_key = "temp123"

UPLOAD_FOLDER = "uploads"
FRAMES_FOLDER = "frames"
RESULTS_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return redirect('/index')

@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if "video" not in request.files:
                return jsonify({"message": "No video uploaded"}), 400

            video = request.files["video"]
            if video.filename == '':
                return jsonify({"message": "No video selected"}), 400

            frame_rate = int(request.form.get("frame_rate", 5))
            video_name = os.path.splitext(video.filename)[0]
            video_path = os.path.join(UPLOAD_FOLDER, video.filename)
            video.save(video_path)

            # Create frames folder
            save_folder = os.path.join(FRAMES_FOLDER, video_name)
            if os.path.exists(save_folder):
                for filename in os.listdir(save_folder):
                    os.remove(os.path.join(save_folder, filename))
                os.rmdir(save_folder)
            os.makedirs(save_folder, exist_ok=True)

            # Extract frames
            frame_files = extract_frames(video_path, frame_rate, save_folder)
            
            # Generate dummy frame IDs (not used in database)
            frame_ids = list(range(len(frame_files)))

            # Run deepfake detection
            summary, total_faces, frame_statuses = run_deepfake_detection(save_folder, frame_ids)

            return jsonify({
                "message": "Video processed successfully!",
                "video_name": video_name,
                "summary": summary,
                "frame_statuses": frame_statuses
            })

        except Exception as e:
            return jsonify({"message": f"Error processing video: {str(e)}"}), 500

    return render_template("index.html")

@app.route('/detailed_analysis/<video_name>')
def get_detailed_analysis(video_name):
    """Endpoint to get detailed analysis results"""
    detailed_result_path = os.path.join('results', f'{video_name}_detailed_analysis.json')
    if os.path.exists(detailed_result_path):
        with open(detailed_result_path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Detailed analysis not found"})

@app.route('/uploads/videos/<filename>')
def serve_uploaded_video(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/uploads')
def uploads():
    # List all uploaded videos without database
    videos = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            videos.append({
                'filename': filename,
                'upload_date': 'N/A',
                'video_id': hash(filename)
            })
    return render_template('uploads.html', videos=videos)

@app.route('/delete_video/<filename>', methods=['POST'])
def delete_video(filename):
    try:
        # Delete video file
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(video_path):
            os.remove(video_path)

        # Delete frames folder
        frame_folder = os.path.join(FRAMES_FOLDER, os.path.splitext(filename)[0])
        if os.path.exists(frame_folder):
            for f in os.listdir(frame_folder):
                os.remove(os.path.join(frame_folder, f))
            os.rmdir(frame_folder)
        
        # Delete results file
        result_json_path = os.path.join(RESULTS_FOLDER, f'{os.path.splitext(filename)[0]}_frame_status.json')
        if os.path.exists(result_json_path):
            os.remove(result_json_path)
            
        # Delete detailed analysis
        detailed_result_path = os.path.join(RESULTS_FOLDER, f'{os.path.splitext(filename)[0]}_detailed_analysis.json')
        if os.path.exists(detailed_result_path):
            os.remove(detailed_result_path)

        return redirect('/uploads')
    except Exception as e:
        return f"Error deleting video: {str(e)}", 500

@app.route("/view_frames/<video_name>")
def view_frames(video_name):
    frame_folder = os.path.join(FRAMES_FOLDER, video_name)
    if not os.path.exists(frame_folder):
        return jsonify([])

    frame_files = sorted(os.listdir(frame_folder))
    frame_urls = [f"/frames/{video_name}/{filename}" for filename in frame_files if filename.endswith(".jpg")]
    return jsonify(frame_urls)

@app.route('/frame_statuses/<video_name>')
def frame_statuses(video_name):
    # Return classification statuses for each frame
    result_path = os.path.join(RESULTS_FOLDER, f'{video_name}_frame_status.json')
    if os.path.exists(result_path):
        with open(result_path) as f:
            return jsonify(json.load(f))
    return jsonify({})

@app.route('/frames/<video_name>/<filename>')
def serve_frame(video_name, filename):
    return send_from_directory(os.path.join(FRAMES_FOLDER, video_name), filename)

def extract_frames(video_path, frame_rate, save_folder):
    cap = cv2.VideoCapture(video_path)
    count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = max(1, fps // frame_rate)
    frame_files = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if count % interval == 0:
            frame_filename = f"frame_{count}.jpg"
            frame_path = os.path.join(save_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_files.append(frame_filename)
        count += 1

    cap.release()
    return frame_files

if __name__ == "__main__":
    app.run(debug=True)
