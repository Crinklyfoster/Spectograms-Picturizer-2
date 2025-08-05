"""
Batch processing module for handling multiple audio files.
Processes files sequentially and organizes results by file.
"""

import os
import json
import time
from backend.spectrograms import generate_all_spectrograms
from backend.features import extract_all_features
from backend.utils import get_upload_path

class BatchProcessor:
    def __init__(self, session_id, file_list):
        self.session_id = session_id
        self.file_list = file_list
        self.results_dir = os.path.join('results', session_id)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def process_batch(self):
        """Process all files in the batch."""
        from app import batch_status
        
        try:
            for i, file_info in enumerate(self.file_list):
                original_filename = file_info['original_name']
                saved_filename = file_info['saved_name']
                
                # Update status
                batch_status[self.session_id]['current_file'] = i + 1
                batch_status[self.session_id]['status'] = 'processing'
                batch_status[self.session_id]['current_filename'] = original_filename
                
                try:
                    self.process_single_file(saved_filename, original_filename)
                    batch_status[self.session_id]['completed_files'].append(original_filename)
                    
                except Exception as e:
                    error_msg = f"Error processing {original_filename}: {str(e)}"
                    batch_status[self.session_id]['errors'].append(error_msg)
                    print(f"Batch processing error: {error_msg}")
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
            
            # Mark as completed
            batch_status[self.session_id]['status'] = 'completed'
            batch_status[self.session_id]['end_time'] = time.time()
            
        except Exception as e:
            batch_status[self.session_id]['status'] = 'error'
            batch_status[self.session_id]['error'] = str(e)
    
    def process_single_file(self, saved_filename, original_filename):
        """Process a single audio file."""
        # Get file path
        audio_path = get_upload_path(saved_filename, self.session_id)
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Create directory for this file's results
        file_id = saved_filename.split('.')[0]
        file_results_dir = os.path.join(self.results_dir, file_id)
        os.makedirs(file_results_dir, exist_ok=True)
        
        # Generate spectrograms in the file's directory
        spectrograms = generate_all_spectrograms(audio_path, self.session_id, file_id)
        
        # Extract features
        features_df = extract_all_features(audio_path)
        features_dict = features_df.to_dict('records')[0]
        
        # Add metadata
        features_dict['filename'] = original_filename
        features_dict['file_id'] = file_id
        features_dict['session_id'] = self.session_id
        
        # Save features as JSON in the file's directory
        features_path = os.path.join(file_results_dir, 'features.json')
        with open(features_path, 'w') as f:
            json.dump(features_dict, f, indent=2, default=str)
        
        return {
            'filename': original_filename,
            'file_id': file_id,
            'spectrograms': spectrograms,
            'features': features_dict
        }
