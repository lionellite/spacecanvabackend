from flask import Blueprint, request, jsonify
from auth import require_auth, require_workspace_access
from database import (
    create_workspace, 
    get_user_workspaces, 
    save_analysis_result,
    get_workspace_analyses
)
import json

workspace_bp = Blueprint('workspace', __name__)

@workspace_bp.route('/api/workspace/create', methods=['POST'])
@require_auth
def create_user_workspace():
    """
    Create a new workspace for the authenticated user
    
    Request body:
    {
        "name": "My Workspace",
        "description": "Optional description"
    }
    """
    data = request.get_json()
    
    if not data or 'name' not in data:
        return jsonify({
            'error': 'Missing required field',
            'message': 'Please provide a workspace name'
        }), 400
    
    name = data.get('name')
    description = data.get('description')
    
    try:
        workspace = create_workspace(
            clerk_user_id=request.clerk_user_id,
            name=name,
            description=description
        )
        
        return jsonify({
            'success': True,
            'workspace': workspace,
            'message': 'Workspace created successfully'
        }), 201
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to create workspace',
            'message': str(e)
        }), 500

@workspace_bp.route('/api/workspace/list', methods=['GET'])
@require_auth
def list_user_workspaces():
    """
    Get all workspaces for the authenticated user
    """
    try:
        workspaces = get_user_workspaces(request.clerk_user_id)
        
        return jsonify({
            'success': True,
            'workspaces': workspaces,
            'count': len(workspaces)
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch workspaces',
            'message': str(e)
        }), 500

@workspace_bp.route('/api/workspace/verify', methods=['POST'])
@require_auth
@require_workspace_access
def verify_workspace():
    """
    Verify access to a workspace
    
    Request body:
    {
        "workspace_key": "your-workspace-key"
    }
    """
    return jsonify({
        'success': True,
        'workspace': request.workspace,
        'message': 'Access granted'
    }), 200

@workspace_bp.route('/api/workspace/analyze', methods=['POST'])
@require_auth
@require_workspace_access
def analyze_in_workspace():
    """
    Run an analysis in a workspace and save the result
    
    Request body:
    {
        "workspace_key": "your-workspace-key",
        "analysis_type": "exoplanet_detection",
        "data": {
            "period": 3.52,
            "duration": 2.5,
            ...
        }
    }
    """
    data = request.get_json()
    
    if not data or 'analysis_type' not in data or 'data' not in data:
        return jsonify({
            'error': 'Missing required fields',
            'message': 'Please provide analysis_type and data'
        }), 400
    
    analysis_type = data.get('analysis_type')
    analysis_data = data.get('data')
    
    try:
        # Run the analysis based on type
        if analysis_type == 'exoplanet_detection':
            # Import the prediction logic
            from app import predict_exoplanet_internal
            
            result = predict_exoplanet_internal(analysis_data)
            
            # Save to database
            analysis_id = save_analysis_result(
                workspace_id=request.workspace['id'],
                analysis_type=analysis_type,
                input_data=json.dumps(analysis_data),
                output_data=json.dumps(result)
            )
            
            return jsonify({
                'success': True,
                'analysis_id': analysis_id,
                'result': result,
                'message': 'Analysis completed and saved'
            }), 200
        else:
            return jsonify({
                'error': 'Unknown analysis type',
                'message': f'Analysis type "{analysis_type}" is not supported'
            }), 400
            
    except Exception as e:
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e)
        }), 500

@workspace_bp.route('/api/workspace/history', methods=['GET'])
@require_auth
@require_workspace_access
def get_workspace_history():
    """
    Get analysis history for a workspace
    
    Query params:
    - workspace_key: your workspace key
    - limit: number of results (default: 50)
    """
    limit = request.args.get('limit', 50, type=int)
    
    try:
        analyses = get_workspace_analyses(
            workspace_id=request.workspace['id'],
            limit=limit
        )
        
        return jsonify({
            'success': True,
            'workspace': request.workspace,
            'analyses': analyses,
            'count': len(analyses)
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch history',
            'message': str(e)
        }), 500
