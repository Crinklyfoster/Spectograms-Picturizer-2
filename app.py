"""
Motor Fault Detection Flask Application - Batch Processing Only
Handles both single and multiple files through the same batch pipeline.
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

app = Flask(__name__)
app.secret_key = 'motor_fault_detection_batch_only_2025'

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg'}
MAX_FILES_PER_SESSION = 100

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global batch status tracking
batch_status = {}
batch_lock = threading.Lock()

def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main upload page for batch processing."""
    return render_template('index.html', max_files=MAX_FILES_PER_SESSION)

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads - both single and multiple through batch processing."""
    # Get files from both possible input names
    files = request.files.getlist('files')  # Multiple files
    single_file = request.files.get('file')  # Single file
    
    # Combine both into one list
    all_files = []
    if single_file and single_file.filename != '':
        all_files.append(single_file)
    if files:
        all_files.extend([f for f in files if f and f.filename != ''])
    
    # Remove duplicates and filter valid files
    valid_files = []
    seen_names = set()
    
    for file in all_files:
        if file and file.filename != '' and file.filename not in seen_names and allowed_file(file.filename):
            valid_files.append(file)
            seen_names.add(file.filename)
    
    if not valid_files:
        return render_template('index.html', 
                             error="No valid audio files found. Please upload WAV, MP3, FLAC, M4A, or OGG files.",
                             max_files=MAX_FILES_PER_SESSION)
    
    if len(valid_files) > MAX_FILES_PER_SESSION:
        return render_template('index.html', 
                             error=f"Too many files. Maximum {MAX_FILES_PER_SESSION} files allowed per session.",
                             max_files=MAX_FILES_PER_SESSION)
    
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        session['upload_time'] = datetime.now().isoformat()
        session['total_files'] = len(valid_files)
        
        print(f"Processing {len(valid_files)} files in session {session_id}")
        
        # Save uploaded files
        saved_files = save_uploaded_files(valid_files, session_id)
        print(f"Saved files: {[f['original_name'] for f in saved_files]}")
        
        # Initialize batch status
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
        app.logger.error(f"Upload error: {str(e)}")
        print(f"Upload error: {str(e)}")
        return render_template('index.html', 
                             error=f"Error uploading files: {str(e)}",
                             max_files=MAX_FILES_PER_SESSION)

def process_batch_files(session_id, file_list):
    """Process all files in batch mode."""
    try:
        print(f"Starting batch processing for session {session_id}")
        results_dir = os.path.join('results', session_id)
        os.makedirs(results_dir, exist_ok=True)
        
        for i, file_info in enumerate(file_list):
            print(f"Processing file {i+1}/{len(file_list)}: {file_info['original_name']}")
            
            # Update status
            with batch_lock:
                if session_id not in batch_status:
                    print(f"Session {session_id} was cleared, stopping processing")
                    break
                
                batch_status[session_id]['current_file'] = i + 1
                batch_status[session_id]['current_filename'] = file_info['original_name']
            
            try:
                # Get file paths
                original_filename = file_info['original_name']
                saved_filename = file_info['saved_name']
                audio_path = get_upload_path(saved_filename, session_id)
                
                print(f"Audio path: {audio_path}")
                
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
                # Create directory for this file's results
                file_id = saved_filename.split('.')[0]
                file_results_dir = os.path.join(results_dir, file_id)
                os.makedirs(file_results_dir, exist_ok=True)
                
                print(f"Results directory: {file_results_dir}")
                
                # Generate spectrograms
                print(f"Generating spectrograms for {original_filename}")
                spectrograms = generate_all_spectrograms(audio_path, session_id, file_id)
                print(f"Generated {len(spectrograms)} spectrograms")
                
                # Extract features
                print(f"Extracting features for {original_filename}")
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
                
                print(f"Saved features to {features_path}")
                
                # Update completed files
                with batch_lock:
                    if session_id in batch_status:
                        batch_status[session_id]['completed_files'].append(original_filename)
                
                print(f"Completed processing {original_filename}")
                
            except Exception as e:
                error_msg = f"Error processing {file_info['original_name']}: {str(e)}"
                print(error_msg)
                
                with batch_lock:
                    if session_id in batch_status:
                        batch_status[session_id]['errors'].append(error_msg)
            
            # Small delay
            time.sleep(0.1)
        
        # Mark as completed
        with batch_lock:
            if session_id in batch_status:
                batch_status[session_id]['status'] = 'completed'
                batch_status[session_id]['end_time'] = time.time()
                print(f"Batch processing completed for session {session_id}")
        
    except Exception as e:
        error_msg = f"Batch processing error: {str(e)}"
        print(error_msg)
        
        with batch_lock:
            if session_id in batch_status:
                batch_status[session_id]['status'] = 'error'
                batch_status[session_id]['error'] = str(e)

@app.route('/batch_progress')
def batch_progress():
    """Show batch processing progress."""
    if 'session_id' not in session:
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
                         files=session.get('file_names', []))

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
    """Display batch results."""
    if 'session_id' not in session:
        return redirect(url_for('index'))
    
    session_id = session['session_id']
    
    # Check if batch processing is complete
    with batch_lock:
        if session_id in batch_status and batch_status[session_id]['status'] != 'completed':
            return redirect(url_for('batch_progress'))
    
    try:
        # Load batch results
        results_dir = os.path.join('results', session_id)
        processed_files = []
        
        print(f"Loading results from: {results_dir}")
        
        if os.path.exists(results_dir):
            for file_id in os.listdir(results_dir):
                file_dir = os.path.join(results_dir, file_id)
                if os.path.isdir(file_dir):
                    print(f"Processing results for file_id: {file_id}")
                    
                    # Load features to get original filename
                    features_path = os.path.join(file_dir, 'features.json')
                    if os.path.exists(features_path):
                        with open(features_path, 'r') as f:
                            features = json.load(f)
                            original_filename = features.get('filename', file_id)
                        
                        # Get spectrogram info
                        spectrograms = {}
                        spectrogram_types = ['mel', 'cqt', 'log_stft', 'wavelet', 'spectral_kurtosis', 'modulation']
                        
                        for spec_type in spectrogram_types:
                            spec_path = os.path.join(file_dir, f'{spec_type}_spectrogram.png')
                            if os.path.exists(spec_path):
                                spectrograms[spec_type] = {
                                    'name': spec_type.replace('_', ' ').title(),
                                    'path': url_for('serve_result_file', 
                                                  session_id=session_id, 
                                                  file_id=file_id,
                                                  filename=f'{spec_type}_spectrogram.png')
                                }
                        
                        processed_files.append({
                            'filename': original_filename,
                            'file_id': file_id,
                            'spectrograms': spectrograms,
                            'features_count': len(features)
                        })
                        
                        print(f"Added processed file: {original_filename} with {len(spectrograms)} spectrograms")
        
        print(f"Total processed files: {len(processed_files)}")
        
        return render_template('batch_results.html',
                             processed_files=processed_files,
                             total_files=len(processed_files),
                             session_id=session_id)
    
    except Exception as e:
        app.logger.error(f"Results error: {str(e)}")
        print(f"Results error: {str(e)}")
        return render_template('index.html', 
                             error=f"Error loading results: {str(e)}",
                             max_files=MAX_FILES_PER_SESSION)

@app.route('/results/<session_id>/<file_id>/<filename>')
def serve_result_file(session_id, file_id, filename):
    """Serve result files (spectrograms)."""
    try:
        if 'session_id' not in session or session['session_id'] != session_id:
            return "Unauthorized", 403
        
        results_dir = os.path.join(os.getcwd(), 'results', session_id, file_id)
        
        if not os.path.exists(os.path.join(results_dir, filename)):
            print(f"File not found: {os.path.join(results_dir, filename)}")
            return "File not found", 404
        
        return send_from_directory(results_dir, filename)
    
    except Exception as e:
        app.logger.error(f"Error serving file {filename}: {str(e)}")
        return "Error serving file", 500

@app.route('/download/features/<format>')
def download_combined_features(format):
    """Download combined features from all processed files."""
    if 'session_id' not in session:
        return redirect(url_for('index'))
    
    session_id = session['session_id']
    
    # Load all features from files
    features = []
    results_dir = os.path.join('results', session_id)
    
    if os.path.exists(results_dir):
        for file_id in os.listdir(results_dir):
            features_path = os.path.join(results_dir, file_id, 'features.json')
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    features.append(json.load(f))
    
    if not features:
        return redirect(url_for('index'))
    
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
    """Download all spectrograms as ZIP file."""
    if 'session_id' not in session:
        return redirect(url_for('index'))
    
    session_id = session['session_id']
    
    try:
        # Create file list from directory structure
        file_list = []
        results_dir = os.path.join('results', session_id)
        
        if os.path.exists(results_dir):
            for file_id in os.listdir(results_dir):
                features_path = os.path.join(results_dir, file_id, 'features.json')
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        features = json.load(f)
                        file_list.append({
                            'original_name': features.get('filename', file_id),
                            'saved_name': f"{file_id}.wav"
                        })
        
        zip_path = create_zip_download(session_id, file_list)
        
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
    """Clear all session data and files."""
    if 'session_id' in session:
        session_id = session['session_id']
        clear_session_files(session_id)
        
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    
