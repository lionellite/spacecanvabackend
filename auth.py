from functools import wraps
from flask import request, jsonify

def require_auth(f):
    """
    Decorator to require authentication for a route
    Expects X-Clerk-User-Id header with the user ID from frontend
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get user_id from X-Clerk-User-Id header
        user_id = request.headers.get('X-Clerk-User-Id')

        if not user_id:
            return jsonify({
                'error': 'No user ID header',
                'message': 'Please provide X-Clerk-User-Id header with your user ID'
            }), 401

        # Add user_id to request context (trusting frontend)
        request.clerk_user_id = user_id

        return f(*args, **kwargs)

    return decorated_function

def require_workspace_access(f):
    """
    Decorator to require workspace access
    Expects workspace_key in request body or query params
    Must be used after @require_auth
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from database import verify_workspace_access
        
        # Get workspace_key from request
        workspace_key = None
        if request.method == 'GET':
            workspace_key = request.args.get('workspace_key')
        else:
            data = request.get_json()
            workspace_key = data.get('workspace_key') if data else None
        
        if not workspace_key:
            return jsonify({
                'error': 'No workspace key',
                'message': 'Please provide a workspace_key'
            }), 400
        
        # Verify access
        workspace = verify_workspace_access(workspace_key, request.clerk_user_id)
        
        if not workspace:
            return jsonify({
                'error': 'Access denied',
                'message': 'You do not have access to this workspace'
            }), 403
        
        # Add workspace to request context
        request.workspace = workspace
        
        return f(*args, **kwargs)
    
    return decorated_function
