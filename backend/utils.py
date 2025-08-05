"""
Utility functions for file handling, cleanup, and batch operations.
"""

import os
import shutil
import uuid
import zipfile
from werkzeug.utils import secure_filename

def save_uploaded_files(files, session_id):
    """
    Save multiple uploaded files with unique names but preserve original names.
    
    Args:
        files: List of Flask file objects
        session_id: Unique session identifier
    
    Returns:
        list: List of dictionaries with original and saved filenames
    """
    # Create session directory
    session_dir = os.path.join('uploads', session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    saved_files = []
    
    for i, file in enumerate(files):
        # Get original filename
        original_filename = secure_filename(file.filename)
        
        # Generate unique filename for storage
        file_extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        
        # Save file
        file_path = os.path.join(session_dir, unique_filename)
        file.save(file_path)
        
        # Store file information
        saved_files.append({
            'original_name': original_filename,
            'saved_name': unique_filename,
            'file_path': file_path
        })
    
    return saved_files

def save_uploaded_file(file, session_id):
    """
    Save single uploaded file (for backward compatibility).
    """
    files = save_uploaded_files([file], session_id)
    return files[0] if files else None

def get_upload_path(saved_filename, session_id):
    """
    Get the full path to an uploaded file.
    
    Args:
        saved_filename: Name of the saved file
        session_id: Session identifier
    
    Returns:
        str: Full path to the file
    """
    return os.path.join('uploads', session_id, saved_filename)

def create_zip_download(session_id, file_list):
    """
    Create a ZIP file containing all spectrograms organized by file.
    
    Args:
        session_id: Session identifier
        file_list: List of file information dictionaries
    
    Returns:
        str: Path to created ZIP file
    """
    results_dir = os.path.join('results', session_id)
    zip_path = os.path.join(results_dir, 'spectrograms.zip')
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_info in file_list:
            original_filename = file_info['original_name']
            saved_filename = file_info['saved_name']
            file_id = saved_filename.split('.')[0]
            
            # Create folder name based on original filename
            folder_name = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename
            folder_name = secure_filename(folder_name)  # Make it safe for filesystem
            
            file_results_dir = os.path.join(results_dir, file_id)
            
            if os.path.exists(file_results_dir):
                # Add all spectrograms from this file's directory
                for filename in os.listdir(file_results_dir):
                    if filename.endswith('.png'):
                        file_path = os.path.join(file_results_dir, filename)
                        archive_path = os.path.join(folder_name, filename)
                        zipf.write(file_path, archive_path)
    
    return zip_path

def clear_session_files(session_id):
    """
    Remove all files associated with a session.
    
    Args:
        session_id: Session identifier to clear
    """
    # Clear upload directory
    upload_dir = os.path.join('uploads', session_id)
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
    
    # Clear results directory
    results_dir = os.path.join('results', session_id)
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)

# ... (keep other utility functions the same) ...
