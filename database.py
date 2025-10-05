import sqlite3
import os
from datetime import datetime
import secrets

DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'workspace.db')

def init_database():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Workspaces table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workspaces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_key TEXT UNIQUE NOT NULL,
            clerk_user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Workspace sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workspace_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id INTEGER NOT NULL,
            session_name TEXT NOT NULL,
            data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
        )
    ''')
    
    # Workspace files table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workspace_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT,
            file_path TEXT,
            file_size INTEGER,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
        )
    ''')
    
    # Analysis results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id INTEGER NOT NULL,
            session_id INTEGER,
            analysis_type TEXT NOT NULL,
            input_data TEXT,
            output_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id),
            FOREIGN KEY (session_id) REFERENCES workspace_sessions(id)
        )
    ''')
    
    # Training sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            dataset_size INTEGER,
            status TEXT DEFAULT 'pending',
            model_path TEXT,
            metrics TEXT,
            epochs_completed INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
        )
    ''')
    
    # Datasets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            num_samples INTEGER,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")

def generate_workspace_key():
    """Generate a unique workspace key"""
    return secrets.token_urlsafe(32)

def create_workspace(clerk_user_id, name, description=None):
    """Create a new workspace for a user"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    workspace_key = generate_workspace_key()
    
    try:
        cursor.execute('''
            INSERT INTO workspaces (workspace_key, clerk_user_id, name, description)
            VALUES (?, ?, ?, ?)
        ''', (workspace_key, clerk_user_id, name, description))
        
        workspace_id = cursor.lastrowid
        conn.commit()
        
        return {
            'id': workspace_id,
            'workspace_key': workspace_key,
            'name': name,
            'description': description
        }
    except sqlite3.IntegrityError:
        # If key collision (very unlikely), try again
        return create_workspace(clerk_user_id, name, description)
    finally:
        conn.close()

def verify_workspace_access(workspace_key, clerk_user_id):
    """Verify if a user has access to a workspace"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, name, description, created_at, is_active
        FROM workspaces
        WHERE workspace_key = ? AND clerk_user_id = ? AND is_active = 1
    ''', (workspace_key, clerk_user_id))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            'id': result[0],
            'name': result[1],
            'description': result[2],
            'created_at': result[3],
            'is_active': result[4]
        }
    return None

def get_user_workspaces(clerk_user_id):
    """Get all workspaces for a user"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, workspace_key, name, description, created_at, is_active
        FROM workspaces
        WHERE clerk_user_id = ? AND is_active = 1
        ORDER BY created_at DESC
    ''', (clerk_user_id,))
    
    results = cursor.fetchall()
    conn.close()
    
    workspaces = []
    for row in results:
        workspaces.append({
            'id': row[0],
            'workspace_key': row[1],
            'name': row[2],
            'description': row[3],
            'created_at': row[4],
            'is_active': row[5]
        })
    
    return workspaces

def save_analysis_result(workspace_id, analysis_type, input_data, output_data, session_id=None):
    """Save an analysis result"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO analysis_results (workspace_id, session_id, analysis_type, input_data, output_data)
        VALUES (?, ?, ?, ?, ?)
    ''', (workspace_id, session_id, analysis_type, input_data, output_data))
    
    result_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return result_id

def get_workspace_analyses(workspace_id, limit=50):
    """Get analysis history for a workspace"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, analysis_type, input_data, output_data, created_at
        FROM analysis_results
        WHERE workspace_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    ''', (workspace_id, limit))
    
    results = cursor.fetchall()
    conn.close()
    
    analyses = []
    for row in results:
        analyses.append({
            'id': row[0],
            'analysis_type': row[1],
            'input_data': row[2],
            'output_data': row[3],
            'created_at': row[4]
        })
    
    return analyses

def save_training_session(workspace_id, model_name, dataset_size, status='pending'):
    """Save a new training session"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO training_sessions (workspace_id, model_name, dataset_size, status)
        VALUES (?, ?, ?, ?)
    ''', (workspace_id, model_name, dataset_size, status))
    
    training_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return training_id

def update_training_session(training_id, status=None, model_path=None, metrics=None, epochs_completed=None):
    """Update a training session"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    updates = []
    params = []
    
    if status:
        updates.append('status = ?')
        params.append(status)
        if status == 'completed':
            updates.append('completed_at = CURRENT_TIMESTAMP')
    
    if model_path:
        updates.append('model_path = ?')
        params.append(model_path)
    
    if metrics:
        updates.append('metrics = ?')
        params.append(metrics)
    
    if epochs_completed is not None:
        updates.append('epochs_completed = ?')
        params.append(epochs_completed)
    
    if updates:
        params.append(training_id)
        query = f"UPDATE training_sessions SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, params)
        conn.commit()
    
    conn.close()

def save_dataset(workspace_id, name, file_path, file_size, num_samples):
    """Save dataset information"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO datasets (workspace_id, name, file_path, file_size, num_samples)
        VALUES (?, ?, ?, ?, ?)
    ''', (workspace_id, name, file_path, file_size, num_samples))
    
    dataset_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return dataset_id

def get_workspace_datasets(workspace_id):
    """Get all datasets for a workspace"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, name, file_path, file_size, num_samples, uploaded_at
        FROM datasets
        WHERE workspace_id = ?
        ORDER BY uploaded_at DESC
    ''', (workspace_id,))
    
    results = cursor.fetchall()
    conn.close()
    
    datasets = []
    for row in results:
        datasets.append({
            'id': row[0],
            'name': row[1],
            'file_path': row[2],
            'file_size': row[3],
            'num_samples': row[4],
            'uploaded_at': row[5]
        })
    
    return datasets

def get_workspace_training_sessions(workspace_id):
    """Get all training sessions for a workspace"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, model_name, dataset_size, status, model_path, metrics, epochs_completed, created_at, completed_at
        FROM training_sessions
        WHERE workspace_id = ?
        ORDER BY created_at DESC
    ''', (workspace_id,))
    
    results = cursor.fetchall()
    conn.close()
    
    sessions = []
    for row in results:
        sessions.append({
            'id': row[0],
            'model_name': row[1],
            'dataset_size': row[2],
            'status': row[3],
            'model_path': row[4],
            'metrics': row[5],
            'epochs_completed': row[6],
            'created_at': row[7],
            'completed_at': row[8]
        })
    
    return sessions

# Initialize database on import
if not os.path.exists(DATABASE_PATH):
    init_database()
