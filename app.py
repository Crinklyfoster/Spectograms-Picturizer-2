"""
Motor Fault Detection Flask Application - Phase 2: Multi-File Batch Processing
Main application file with routes for batch upload, analysis, and organized downloads.
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
from backend.utils import save_uploaded_files, clear_session_files, get_upload_path, create_zip_download
from backend.batch_processor import BatchProcessor

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

# Global batch status tracking
batch_status = {}

def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main upload page for multiple files."""
    return render_template('index.html', max_files=MAX_FILES_PER_SESSION)

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle multiple file upload and start batch analysis."""
    if 'files' not in request.files:
        return redirect(request.url)
    
    files = request.files.getlist('files')
    
    if not files or all(f.filename == '' for f in files):
        return render_template('index.html', 
                             error="No files selected.", 
                             max_files=MAX_FILES_PER_SESSION)
    
    # Validate file count
    if len(files) > MAX_FILES_PER_SESSION:
        return render_template('index.html', 
                             error=f"Too many files. Maximum {MAX_FILES_PER_SESSION} files allowed per session.",
                             max_files=MAX_FILES_PER_SESSION)
    
    valid_files = []
    for file in files:
        if file and allowed_file(file.filename):
            valid_files.append(file)
        else:
            return render_template('index.html', 
                                 error=f"Invalid file: {file.filename}. Only WAV, MP3, FLAC, M4A, OGG files allowed.",
                                 max_files=MAX_FILES_PER_SESSION)
    
    if not valid_files:
        return render_template('index.html', 
                             error="No valid audio files found.",
                             max_files=MAX_FILES_PER_SESSION)
    
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        session['upload_time'] = datetime.now().isoformat()
        
        # Save uploaded files
        saved_files = save_uploaded_files(valid_files, session_id)
        session['files'] = saved_files
        session['total_files'] = len(saved_files)
        
        # Initialize batch status
        batch_status[session_id] = {
            'status': 'processing',
            'current_file': 0,
            'total_files': len(saved_files),
            'completed_files': [],
            'errors': [],
            'start_time': time.time()
        }
        
        # Start batch processing in background
        processor = BatchProcessor(session_id, saved_files)
        thread = threading.Thread(target=processor.process_batch)
        thread.daemon = True
        thread.start()
        
        return redirect(url_for('batch_progress'))
        
    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        return render_template('index.html', 
                             error="Error uploading files. Please try again.",
                             max_files=MAX_FILES_PER_SESSION)

@app.route('/batch_progress')
def batch_progress():
    """Show batch processing progress."""
    if 'session_id' not in session:
        return redirect(url_for('index'))
    
    session_id = session['session_id']
    
    if session_id not in batch_status:
        return redirect(url_for('index'))
    
    status = batch_status[session_id]
    
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
    
    if session_id not in batch_status:
        return jsonify({'error': 'No batch found'}), 404
    
    status = batch_status[session_id]
    
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
    """Display batch analysis results."""
    if 'session_id' not in session:
        return redirect(url_for('index'))
    
    session_id = session['session_id']
    
    # Check if batch processing is complete
    if session_id in batch_status and batch_status[session_id]['status'] != 'completed':
        return redirect(url_for('batch_progress'))
    
    try:
        # Load batch results
        results_dir = os.path.join('results', session_id)
        
        # Get all processed files
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
                            'path': url_for('serve_result_file', 
                                          session_id=session_id, 
                                          file_id=file_id,
                                          filename=f'{spec_type}_spectrogram.png')
                        }
                
                # Load features
                features_path = os.path.join(file_results_dir, 'features.json')
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        features = json.load(f)
                        features['filename'] = original_filename
                        combined_features.append(features)
                
                processed_files.append({
                    'filename': original_filename,
                    'file_id': file_id,
                    'spectrograms': spectrograms,
                    'features_count': len(features) if 'features' in locals() else 0
                })
        
        # Store combined features for download
        session['combined_features'] = combined_features
        
        return render_template('batch_results.html',
                             processed_files=processed_files,
                             total_files=len(processed_files),
                             session_id=session_id)
    
    except Exception as e:
        app.logger.error(f"Results error: {str(e)}")
        return render_template('index.html', 
                             error=f"Error loading results: {str(e)}",
                             max_files=MAX_FILES_PER_SESSION)

@app.route('/results/<session_id>/<file_id>/<filename>')
def serve_result_file(session_id, file_id, filename):
    """Serve generated result files (spectrograms)."""
    try:
        # Security check
        if 'session_id' not in session or session['session_id'] != session_id:
            return "Unauthorized", 403
        
        results_dir = os.path.join(os.getcwd(), 'results', session_id, file_id)
        
        if not os.path.exists(os.path.join(results_dir, filename)):
            return "File not found", 404
        
        return send_from_directory(results_dir, filename)
    
    except Exception as e:
        app.logger.error(f"Error serving file {filename}: {str(e)}")
        return "Error serving file", 500

@app.route('/download/features/<format>')
def download_combined_features(format):
    """Download combined features from all files in CSV or JSON format."""
    if 'combined_features' not in session:
        return redirect(url_for('index'))
    
    features = session['combined_features']
    session_id = session.get('session_id', 'batch')
    
    if format == 'csv':
        # Create combined CSV
        df = pd.DataFrame(features)
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(output, 
                        mimetype='text/csv',
                        as_attachment=True,
                        download_name=f'batch_features_{session_id[:8]}.csv')
    
    elif format == 'json':
        # Create combined JSON
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
    """Download all spectrograms as a ZIP file with organized folders."""
    if 'session_id' not in session:
        return redirect(url_for('index'))
    
    session_id = session['session_id']
    
    try:
        # Create ZIP file
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
