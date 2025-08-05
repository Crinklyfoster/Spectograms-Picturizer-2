"""
Motor Fault Detection Flask Application - Phase 2: Fixed Hybrid Processing
Supports both single file and batch processing with proper error handling.
"""

from flask import Flask, render_template, request, redirect, url_for, session, send_file, send_from_directory, jsonify
import os
import uuid
import threading
import time
from datetime import datetime
from werkzeug.utils import secure_filename
import pandas as pd
import json
from io import BytesIO

from backend.spectrograms import generate_all_spectrograms
from backend.features import extract_all_features
from backend.utils import save_uploaded_file, save_uploaded_files, clear_session_files, get_upload_path, create_zip_download

app = Flask(__name__)
app.secret_key = 'motor_fault_detection_secret_key_2025_phase2'

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg'}
MAX_FILES_PER_SESSION = 100
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB per file

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global batch status tracking with thread lock
batch_status = {}
batch_lock = threading.Lock()

def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main upload page supporting both single and multiple files."""
    return render_template('index.html', max_files=MAX_FILES_PER_SESSION)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle both single and multiple file uploads."""
    # Check if it's multiple files or single file
    files = request.files.getlist('files')
    single_file = request.files.get('file')
    
    # Handle single file upload (backward compatibility)
    if single_file and single_file.filename != '':
        return handle_single_file_upload(single_file)
    
    # Handle multiple files upload
    elif files and any(f.filename != '' for f in files):
        return handle_batch_upload(files)
    
    else:
        return render_template('index.html', 
                             error="No files selected.", 
                             max_files=MAX_FILES_PER_SESSION)

def handle_single_file_upload(file):
    """Handle single file upload (original functionality)."""
    if not allowed_file(file.filename):
        return render_template('index.html', 
                             error="Invalid file type. Please upload WAV, MP3, FLAC, M4A, or OGG files.",
                             max_files=MAX_FILES_PER_SESSION)
    
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        session['processing_mode'] = 'single'
        
        # Save uploaded file
        file_info = save_uploaded_file(file, session_id)
        
        # Store both original and saved filenames
        session['original_filename'] = file_info['original_name']
        session['saved_filename'] = file_info['saved_name']
        
        return redirect(url_for('results'))
        
    except Exception as e:
        app.logger.error(f"Single file upload error: {str(e)}")
        return render_template('index.html', 
                             error="Error uploading file. Please try again.",
                             max_files=MAX_FILES_PER_SESSION)

def handle_batch_upload(files):
    """Handle multiple files upload."""
    # Filter valid files
    valid_files = [f for f in files if f and f.filename != '' and allowed_file(f.filename)]
    
    if not valid_files:
        return render_template('index.html', 
                             error="No valid audio files found.",
                             max_files=MAX_FILES_PER_SESSION)
    
    if len(valid_files) > MAX_FILES_PER_SESSION:
        return render_template('index.html', 
                             error=f"Too many files. Maximum {MAX_FILES_PER_SESSION} files allowed per session.",
                             max_files=MAX_FILES_PER_SESSION)
    
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        session['processing_mode'] = 'batch'
        session['upload_time'] = datetime.now().isoformat()
        
        # Save uploaded files
        saved_files = save_uploaded_files(valid_files, session_id)
        session['files'] = saved_files
        session['total_files'] = len(saved_files)
        
        # Initialize batch status with thread safety
        with batch_lock:
            batch_status[session_id] = {
                'status': 'processing',
                'current_file': 0,
                'total_files': len(saved_files),
                'completed_files': [],
                'errors': [],
                'start_time': time.time(),
                'current_filename': 'Initializing...'
            }
        
        # Start batch processing in background
        thread = threading.Thread(target=process_batch_files, args=(session_id, saved_files))
        thread.daemon = True
        thread.start()
        
        return redirect(url_for('batch_progress'))
        
    except Exception as e:
        app.logger.error(f"Batch upload error: {str(e)}")
        return render_template('index.html', 
                             error="Error uploading files. Please try again.",
                             max_files=MAX_FILES_PER_SESSION)

def process_batch_files(session_id, file_list):
    """Process batch files in background thread."""
    try:
        results_dir = os.path.join('results', session_id)
        os.makedirs(results_dir, exist_ok=True)
        
        for i, file_info in enumerate(file_list):
            with batch_lock:
                if session_id not in batch_status:
                    break  # Session was cleared
                
                batch_status[session_id]['current_file'] = i + 1
                batch_status[session_id]['current_filename'] = file_info['original_name']
            
            try:
                # Process single file
                original_filename = file_info['original_name']
                saved_filename = file_info['saved_name']
                audio_path = get_upload_path(saved_filename, session_id)
                
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
                # Create directory for this file's results
                file_id = saved_filename.split('.')[0]
                file_results_dir = os.path.join(results_dir, file_id)
                os.makedirs(file_results_dir, exist_ok=True)
                
                # Generate spectrograms
                generate_all_spectrograms(audio_path, session_id, file_id)
                
                # Extract features
                features_df = extract_all_features(audio_path)
                features_dict = features_df.to_dict('records')[0]
                
                # Add metadata
                features_dict['filename'] = original_filename
                features_dict['file_id'] = file_id
                features_dict['session_id'] = session_id
                
                # Save features as JSON
                features_path = os.path.join(file_results_dir, 'features.json')
                with open(features_path, 'w') as f:
                    json.dump(features_dict, f, indent=2, default=str)
                
                # Update completed files
                with batch_lock:
                    if session_id in batch_status:
                        batch_status[session_id]['completed_files'].append(original_filename)
                
            except Exception as e:
                error_msg = f"Error processing {file_info['original_name']}: {str(e)}"
                with batch_lock:
                    if session_id in batch_status:
                        batch_status[session_id]['errors'].append(error_msg)
                app.logger.error(error_msg)
            
            # Small delay to prevent overwhelming
            time.sleep(0.1)
        
        # Mark as completed
        with batch_lock:
            if session_id in batch_status:
                batch_status[session_id]['status'] = 'completed'
                batch_status[session_id]['end_time'] = time.time()
        
    except Exception as e:
        with batch_lock:
            if session_id in batch_status:
                batch_status[session_id]['status'] = 'error'
                batch_status[session_id]['error'] = str(e)
        app.logger.error(f"Batch processing error: {str(e)}")

@app.route('/batch_progress')
def batch_progress():
    """Show batch processing progress."""
    if 'session_id' not in session or session.get('processing_mode') != 'batch':
        return redirect(url_for('index'))
    
    session_id = session['session_id']
    
    with batch_lock:
        if session_id not in batch_status:
            return redirect(url_for('index'))
        status = batch_status[session_id].copy()
    
    if status['status'] == 'completed':
        return redirect(url_for('results'))
    
    return render_template('batch_progress.html', 
                         status=status,
                         files=session.get('files', []))

@app.route('/batch_status')
def get_batch_status():
    """API endpoint for batch processing status."""
    if 'session_id' not in session:
        return jsonify({'error': 'No session found'}), 400
    
    session_id = session['session_id']
    
    with batch_lock:
        if session_id not in batch_status:
            return jsonify({'error': 'No batch found'}), 404
        status = batch_status[session_id].copy()
    
    # Calculate progress percentage
    progress = (status['current_file'] / status['total_files']) * 100 if status['total_files'] > 0 else 0
    
    response = {
        'status': status['status'],
        'current_file': status['current_file'],
        'total_files': status['total_files'],
        'progress': round(progress, 1),
        'completed_files': status['completed_files'],
        'errors': status['errors'],
        'current_filename': status.get('current_filename', '')
    }
    
    return jsonify(response)

@app.route('/results')
def results():
    """Display results for both single and batch processing."""
    if 'session_id' not in session:
        return redirect(url_for('index'))
    
    session_id = session['session_id']
    processing_mode = session.get('processing_mode', 'single')
    
    if processing_mode == 'batch':
        return handle_batch_results()
    else:
        return handle_single_results()

def handle_single_results():
    """Handle single file results (original functionality)."""
    session_id = session['session_id']
    saved_filename = session.get('saved_filename')
    original_filename = session.get('original_filename', saved_filename)
    
    if not saved_filename:
        return redirect(url_for('index'))
    
    try:
        # Get file path using saved filename
        audio_path = get_upload_path(saved_filename, session_id)
        
        if not os.path.exists(audio_path):
            return redirect(url_for('index'))
        
        # Generate spectrograms (single file mode)
        spectrogram_paths = generate_all_spectrograms(audio_path, session_id)
        
        # Convert file paths to web URLs for each spectrogram
        for spec_type in spectrogram_paths:
            if 'path' in spectrogram_paths[spec_type]:
                filename_only = os.path.basename(spectrogram_paths[spec_type]['path'])
                spectrogram_paths[spec_type]['path'] = url_for('serve_result_file', 
                                                              session_id=session_id, 
                                                              filename=filename_only)
        
        # Extract features
        features_df = extract_all_features(audio_path)
        
        # Store features in session for download
        features_dict = features_df.to_dict('records')[0]
        features_dict['original_filename'] = original_filename
        session['features'] = features_dict
        
        # Convert features to readable format for display
        features_display = {}
        for key, value in session['features'].items():
            if isinstance(value, float):
                features_display[key] = round(value, 4)
            else:
                features_display[key] = value
        
        return render_template('results.html', 
                             spectrograms=spectrogram_paths,
                             features=features_display,
                             filename=original_filename)
    
    except Exception as e:
        app.logger.error(f"Single file results error: {str(e)}")
        return render_template('index.html', 
                             error=f"Error loading results: {str(e)}",
                             max_files=MAX_FILES_PER_SESSION)

def handle_batch_results():
    """Handle batch processing results."""
    session_id = session['session_id']
    
    # Check if batch processing is complete
    with batch_lock:
        if session_id in batch_status and batch_status[session_id]['status'] != 'completed':
            return redirect(url_for('batch_progress'))
    
    try:
        # Load batch results
        results_dir = os.path.join('results', session_id)
        processed_files = []
        combined_features = []
        
        for file_info in session.get('files', []):
            original_filename = file_info['original_name']
            saved_filename = file_info['saved_name']
            file_id = saved_filename.split('.')[0]
            file_results_dir = os.path.join(results_dir, file_id)
            
            if os.path.exists(file_results_dir):
                # Get spectrogram info
                spectrograms = {}
                spectrogram_types = ['mel', 'cqt', 'log_stft', 'wavelet', 'spectral_kurtosis', 'modulation']
                
                for spec_type in spectrogram_types:
                    spec_path = os.path.join(file_results_dir, f'{spec_type}_spectrogram.png')
                    if os.path.exists(spec_path):
                        spectrograms[spec_type] = {
                            'name': spec_type.replace('_', ' ').title(),
                            'path': url_for('serve_batch_result_file', 
                                          session_id=session_id, 
                                          file_id=file_id,
                                          filename=f'{spec_type}_spectrogram.png')
                        }
                
                # Load features
                features_path = os.path.join(file_results_dir, 'features.json')
                features = {}
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        features = json.load(f)
                        features['filename'] = original_filename
                        combined_features.append(features)
                
                processed_files.append({
                    'filename': original_filename,
                    'file_id': file_id,
                    'spectrograms': spectrograms,
                    'features_count': len(features)
                })
        
        # Store combined features for download
        session['combined_features'] = combined_features
        
        return render_template('batch_results.html',
                             processed_files=processed_files,
                             total_files=len(processed_files),
                             session_id=session_id)
    
    except Exception as e:
        app.logger.error(f"Batch results error: {str(e)}")
        return render_template('index.html', 
                             error=f"Error loading results: {str(e)}",
                             max_files=MAX_FILES_PER_SESSION)

@app.route('/results/<session_id>/<filename>')
def serve_result_file(session_id, filename):
    """Serve single file results."""
    try:
        if 'session_id' not in session or session['session_id'] != session_id:
            return "Unauthorized", 403
        
        results_dir = os.path.join(os.getcwd(), 'results', session_id)
        
        if not os.path.exists(os.path.join(results_dir, filename)):
            return "File not found", 404
        
        return send_from_directory(results_dir, filename)
    
    except Exception as e:
        app.logger.error(f"Error serving single file {filename}: {str(e)}")
        return "Error serving file", 500

@app.route('/results/<session_id>/<file_id>/<filename>')
def serve_batch_result_file(session_id, file_id, filename):
    """Serve batch file results."""
    try:
        if 'session_id' not in session or session['session_id'] != session_id:
            return "Unauthorized", 403
        
        results_dir = os.path.join(os.getcwd(), 'results', session_id, file_id)
        
        if not os.path.exists(os.path.join(results_dir, filename)):
            return "File not found", 404
        
        return send_from_directory(results_dir, filename)
    
    except Exception as e:
        app.logger.error(f"Error serving batch file {filename}: {str(e)}")
        return "Error serving file", 500

@app.route('/download/<format>')
def download_features(format):
    """Download features for single file."""
    if 'features' not in session:
        return redirect(url_for('index'))
    
    features = session['features']
    original_filename = session.get('original_filename', 'motor_features')
    base_name = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename
    
    if format == 'csv':
        df = pd.DataFrame([features])
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(output, 
                        mimetype='text/csv',
                        as_attachment=True,
                        download_name=f'{base_name}_features.csv')
    
    elif format == 'json':
        output = BytesIO()
        json_str = json.dumps(features, indent=2)
        output.write(json_str.encode())
        output.seek(0)
        
        return send_file(output,
                        mimetype='application/json',
                        as_attachment=True,
                        download_name=f'{base_name}_features.json')
    
    return redirect(url_for('results'))

@app.route('/download/features/<format>')
def download_combined_features(format):
    """Download combined features from batch processing."""
    if 'combined_features' not in session:
        return redirect(url_for('index'))
    
    features = session['combined_features']
    session_id = session.get('session_id', 'batch')
    
    if format == 'csv':
        df = pd.DataFrame(features)
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(output, 
                        mimetype='text/csv',
                        as_attachment=True,
                        download_name=f'batch_features_{session_id[:8]}.csv')
    
    elif format == 'json':
        output = BytesIO()
        json_str = json.dumps(features, indent=2, default=str)
        output.write(json_str.encode())
        output.seek(0)
        
        return send_file(output,
                        mimetype='application/json',
                        as_attachment=True,
                        download_name=f'batch_features_{session_id[:8]}.json')
    
    return redirect(url_for('results'))

@app.route('/download/spectrograms')
def download_spectrograms_zip():
    """Download all spectrograms as ZIP."""
    if 'session_id' not in session:
        return redirect(url_for('index'))
    
    session_id = session['session_id']
    
    try:
        zip_path = create_zip_download(session_id, session.get('files', []))
        
        if not os.path.exists(zip_path):
            return "Error creating ZIP file", 500
        
        return send_file(zip_path,
                        mimetype='application/zip',
                        as_attachment=True,
                        download_name=f'spectrograms_{session_id[:8]}.zip')
    
    except Exception as e:
        app.logger.error(f"Error creating ZIP download: {str(e)}")
        return "Error creating download", 500

@app.route('/clear', methods=['POST'])
def clear_results():
    """Clear all session data and uploaded files."""
    if 'session_id' in session:
        session_id = session['session_id']
        clear_session_files(session_id)
        
        # Clean up batch status
        with batch_lock:
            if session_id in batch_status:
                del batch_status[session_id]
    
    session.clear()
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html', 
                         error="Page not found.",
                         max_files=MAX_FILES_PER_SESSION), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal server error: {str(error)}")
    return render_template('index.html', 
                         error="Internal server error.",
                         max_files=MAX_FILES_PER_SESSION), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
