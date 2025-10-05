# 🔐 Expert Workspace API Documentation

## 🎯 Overview

The Expert Workspace API provides authenticated access to personal workspaces for exoplanet analysis with Clerk authentication.

## 🔑 Authentication

All workspace endpoints require Clerk authentication via the `Authorization` header:

```
Authorization: Bearer <clerk_session_token>
```

Get the session token from Clerk's `useAuth()` hook in your frontend.

## 📡 Endpoints

### 1. Create Workspace

**POST** `/api/workspace/create`

Create a new workspace for the authenticated user.

**Headers:**
```
Authorization: Bearer <clerk_session_token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "name": "My Exoplanet Research",
  "description": "Workspace for analyzing Kepler candidates"
}							

```

**Response (201):**
```json
{
  "success": true,
  "workspace": {
    "id": 1,
    "workspace_key": "abc123...",
    "name": "My Exoplanet Research",
    "description": "Workspace for analyzing Kepler candidates"
  },
  "message": "Workspace created successfully"
}
```

### 2. List Workspaces

**GET** `/api/workspace/list`

Get all workspaces for the authenticated user.

**Headers:**
```
Authorization: Bearer <clerk_session_token>
```

**Response (200):**
```json
{
  "success": true,
  "workspaces": [
    {
      "id": 1,
      "workspace_key": "abc123...",
      "name": "My Exoplanet Research",
      "description": "Workspace for analyzing Kepler candidates",
      "created_at": "2025-01-05T08:00:00",
      "is_active": 1
    }
  ],
  "count": 1
}
```

### 3. Verify Workspace Access

**POST** `/api/workspace/verify`

Verify that you have access to a specific workspace.

**Headers:**
```
Authorization: Bearer <clerk_session_token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "workspace_key": "abc123..."
}
```

**Response (200):**
```json
{
  "success": true,
  "workspace": {
    "id": 1,
    "name": "My Exoplanet Research",
    "description": "Workspace for analyzing Kepler candidates",
    "created_at": "2025-01-05T08:00:00",
    "is_active": 1
  },
  "message": "Access granted"
}
```

**Response (403):**
```json
{
  "error": "Access denied",
  "message": "You do not have access to this workspace"
}
```

### 4. Run Analysis in Workspace

**POST** `/api/workspace/analyze`

Run an exoplanet analysis and save the result to your workspace.

**Headers:**
```
Authorization: Bearer <clerk_session_token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "workspace_key": "abc123...",
  "analysis_type": "exoplanet_detection",
  "data": {
    "period": 3.52,
    "duration": 2.5,
    "depth": 1000,
    "impact": 0.5,
    "snr": 15.2,
    "steff": 5778,
    "srad": 1.0,
    "slogg": 4.5,
    "tmag": 12.5
  }
}
```

**Response (200):**
```json
{
  "success": true,
  "analysis_id": 42,
  "result": {
    "prediction": {
      "label": "CONFIRMED",
      "classification": "Confirmed Exoplanet",
      "confidence": 0.95,
      ...
    },
    ...
  },
  "message": "Analysis completed and saved"
}
```

### 5. Get Workspace History

**GET** `/api/workspace/history?workspace_key=abc123&limit=50`

Get analysis history for a workspace.

**Headers:**
```
Authorization: Bearer <clerk_session_token>
```

**Query Parameters:**
- `workspace_key` (required): Your workspace key
- `limit` (optional): Number of results (default: 50)

**Response (200):**
```json
{
  "success": true,
  "workspace": {
    "id": 1,
    "name": "My Exoplanet Research",
    ...
  },
  "analyses": [
    {
      "id": 42,
      "analysis_type": "exoplanet_detection",
      "input_data": "{...}",
      "output_data": "{...}",
      "created_at": "2025-01-05T10:30:00"
    }
  ],
  "count": 1
}
```

## 🔒 Security

### Clerk Integration

1. **Session Tokens**: All requests must include a valid Clerk session token
2. **User Verification**: The backend verifies the token with Clerk's API
3. **Workspace Keys**: Each workspace has a unique key tied to the user's Clerk ID
4. **Access Control**: Users can only access their own workspaces

### Development Mode

If `CLERK_SECRET_KEY` is not set in `.env`, the system runs in development mode:
- Authentication is bypassed
- The token is used directly as the user ID
- **⚠️ DO NOT use in production!**

## 🧪 Testing

### 1. Create a workspace

```bash
curl -X POST http://localhost:5000/api/workspace/create \
  -H "Authorization: Bearer test_user_123" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Workspace",
    "description": "My test workspace"
  }'
```

### 2. List workspaces

```bash
curl -X GET http://localhost:5000/api/workspace/list \
  -H "Authorization: Bearer test_user_123"
```

### 3. Run analysis

```bash
curl -X POST http://localhost:5000/api/workspace/analyze \
  -H "Authorization: Bearer test_user_123" \
  -H "Content-Type: application/json" \
  -d '{
    "workspace_key": "YOUR_WORKSPACE_KEY",
    "analysis_type": "exoplanet_detection",
    "data": {
      "period": 3.52,
      "duration": 2.5,
      "depth": 1000,
      "impact": 0.5,
      "snr": 15.2,
      "steff": 5778,
      "srad": 1.0,
      "slogg": 4.5,
      "tmag": 12.5
    }
  }'
```

## 📊 Database Schema

### workspaces
- `id`: Primary key
- `workspace_key`: Unique workspace identifier
- `clerk_user_id`: Clerk user ID (from authentication)
- `name`: Workspace name
- `description`: Optional description
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp
- `is_active`: Active status

### analysis_results
- `id`: Primary key
- `workspace_id`: Foreign key to workspaces
- `analysis_type`: Type of analysis
- `input_data`: JSON input data
- `output_data`: JSON output data
- `created_at`: Analysis timestamp

## 🚀 Frontend Integration

```typescript
// Get Clerk session token
const { getToken } = useAuth();
const token = await getToken();

// Create workspace
const response = await fetch('http://localhost:5000/api/workspace/create', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    name: 'My Workspace',
    description: 'Research workspace'
  })
});

const data = await response.json();
console.log('Workspace created:', data.workspace);
```
