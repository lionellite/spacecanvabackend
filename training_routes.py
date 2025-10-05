from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from auth import require_auth, require_workspace_access
from database import (
    save_dataset,
    get_workspace_datasets,
    get_workspace_training_sessions
)
from training import prepare_dataset_from_csv, train_model
import os
import threading

training_bp = Blueprint('training', __name__)

# Upload configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@training_bp.route('/api/training/upload-dataset', methods=['POST'])
@require_auth
@require_workspace_access
def upload_dataset():
    """
    Upload a CSV dataset for training
    
    Form data:
    - file: CSV file
    - workspace_key: workspace key
    - name: dataset name (optional)
    """
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file provided',
            'message': 'Please upload a CSV file'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'error': 'No file selected',
            'message': 'Please select a file'
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Only CSV files are allowed'
        }), 400
    
    try:
        # Secure filename
        filename = secure_filename(file.filename)
        dataset_name = request.form.get('name', filename)
        
        # Save file
        file_path = os.path.join(UPLOAD_FOLDER, f"{request.workspace['id']}_{filename}")
        file.save(file_path)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Quick validation - count lines
        with open(file_path, 'r') as f:
            num_samples = sum(1 for line in f) - 1  # Exclude header
        
        # Save to database
        dataset_id = save_dataset(
            workspace_id=request.workspace['id'],
            name=dataset_name,
            file_path=file_path,
            file_size=file_size,
            num_samples=num_samples
        )
        
        return jsonify({
            'success': True,
            'dataset': {
                'id': dataset_id,
                'name': dataset_name,
                'file_size': file_size,
                'num_samples': num_samples
            },
            'message': 'Dataset uploaded successfully'
        }), 201
        
    except Exception as e:
        return jsonify({
            'error': 'Upload failed',
            'message': str(e)
        }), 500

@training_bp.route('/api/training/datasets', methods=['GET'])
@require_auth
@require_workspace_access
def list_datasets():
    """
    Get all datasets for a workspace
    
    Query params:
    - workspace_key: workspace key
    """
    try:
        datasets = get_workspace_datasets(request.workspace['id'])
        
        return jsonify({
            'success': True,
            'datasets': datasets,
            'count': len(datasets)
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch datasets',
            'message': str(e)
        }), 500

@training_bp.route('/api/training/start', methods=['POST'])
@require_auth
@require_workspace_access
def start_training():
    """
    Start training a model on a dataset
    
    Request body:
    {
        "workspace_key": "your-workspace-key",
        "dataset_id": 1,
        "model_name": "my_model",
        "epochs": 50
    }
    """
    data = request.get_json()
    
    if not data or 'dataset_id' not in data:
        return jsonify({
            'error': 'Missing required field',
            'message': 'Please provide dataset_id'
        }), 400
    
    dataset_id = data.get('dataset_id')
    model_name = data.get('model_name', 'exoplanet_model')
    epochs = data.get('epochs', 50)
    
    try:
        # Get dataset
        datasets = get_workspace_datasets(request.workspace['id'])
        dataset_info = next((d for d in datasets if d['id'] == dataset_id), None)
        
        if not dataset_info:
            return jsonify({
                'error': 'Dataset not found',
                'message': f'Dataset {dataset_id} not found in this workspace'
            }), 404
        
        # Start training in background thread
        def train_async():
            try:
                # Prepare dataset
                dataset = prepare_dataset_from_csv(dataset_info['file_path'])
                
                # Train model
                train_model(
                    dataset=dataset,
                    workspace_id=request.workspace['id'],
                    model_name=model_name,
                    epochs=epochs
                )
            except Exception as e:
                print(f"❌ Training failed: {e}")
        
        thread = threading.Thread(target=train_async)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Training started in background',
            'dataset': dataset_info,
            'model_name': model_name,
            'epochs': epochs
        }), 202  # Accepted
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to start training',
            'message': str(e)
        }), 500

@training_bp.route('/api/training/sessions', methods=['GET'])
@require_auth
@require_workspace_access
def list_training_sessions():
    """
    Get all training sessions for a workspace
    
    Query params:
    - workspace_key: workspace key
    """
    try:
        sessions = get_workspace_training_sessions(request.workspace['id'])
        
        return jsonify({
            'success': True,
            'sessions': sessions,
            'count': len(sessions)
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch training sessions',
            'message': str(e)
        }), 500

@training_bp.route('/api/training/session/<int:session_id>', methods=['GET'])
@require_auth
def get_training_session(session_id):
    """
    Get details of a specific training session
    """
    try:
        sessions = get_workspace_training_sessions(request.workspace['id'])
        session = next((s for s in sessions if s['id'] == session_id), None)
        
        if not session:
            return jsonify({
                'error': 'Session not found',
                'message': f'Training session {session_id} not found'
            }), 404
        
        return jsonify({
            'success': True,
            'session': session
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch session',
            'message': str(e)
        }), 500
