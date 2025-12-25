# JWT Authentication API

## Overview

This guide covers the JWT authentication system added to your LLM Chat application.

## Features

- **User Registration** - Create new user accounts
- **User Login** - Authenticate with username/password
- **Token Refresh** - Extend session without re-logging in
- **Protected Routes** - Secure API endpoints
- **User Preferences** - Store per-user settings
- **Automatic Token Expiry** - Tokens expire for security

## Authentication Flow

```
1. User Registration
   POST /auth/register → Create account

2. User Login
   POST /auth/login → Get JWT + Refresh Token

3. Use Token
   GET /api/chat (with Authorization header) → Authenticated request

4. Token Expiry
   Refresh token expires after 30 days
   Access token expires after 7 days

5. Refresh Token
   POST /auth/refresh → Get new access token
```

## API Endpoints

### 1. Register User

**Endpoint:** `POST /auth/register`

**Request:**
```json
{
  "username": "john_doe",
  "password": "secure_password_123",
  "email": "john@example.com"
}
```

**Response (201):**
```json
{
  "success": true,
  "message": "User registered successfully",
  "user": {
    "username": "john_doe",
    "email": "john@example.com",
    "preferences": {
      "theme": "dark",
      "defaultModel": "gpt-oss-20",
      "temperature": 0.7,
      "maxTokens": 2000
    }
  }
}
```

**Errors:**
- 400: Username already exists
- 400: Password too short (min 8 characters)
- 500: Server error

### 2. Login User

**Endpoint:** `POST /auth/login`

**Request:**
```json
{
  "username": "john_doe",
  "password": "secure_password_123"
}
```

**Response (200):**
```json
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refreshToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "username": "john_doe",
    "email": "john@example.com",
    "preferences": {
      "theme": "dark",
      "defaultModel": "gpt-oss-20",
      "temperature": 0.7,
      "maxTokens": 2000
    }
  }
}
```

**Errors:**
- 400: Username/password required
- 401: User not found
- 401: Invalid password
- 500: Server error

### 3. Refresh Token

**Endpoint:** `POST /auth/refresh`

**Request:**
```json
{
  "refreshToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response (200):**
```json
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Errors:**
- 400: Refresh token required
- 401: Invalid token
- 401: Token expired
- 500: Server error

### 4. Get User Profile

**Endpoint:** `GET /auth/profile`

**Headers:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response (200):**
```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "preferences": {
    "theme": "dark",
    "defaultModel": "gpt-oss-20",
    "temperature": 0.7,
    "maxTokens": 2000
  },
  "createdAt": "2025-12-22T10:30:00.000Z"
}
```

**Errors:**
- 401: Missing authorization header
- 401: Invalid token
- 404: User not found
- 500: Server error

### 5. Update User Preferences

**Endpoint:** `PUT /auth/preferences`

**Headers:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Request:**
```json
{
  "preferences": {
    "theme": "light",
    "defaultModel": "llama3.2",
    "temperature": 0.5,
    "maxTokens": 3000
  }
}
```

**Response (200):**
```json
{
  "success": true,
  "preferences": {
    "theme": "light",
    "defaultModel": "llama3.2",
    "temperature": 0.5,
    "maxTokens": 3000
  }
}
```

**Errors:**
- 400: Preferences required
- 401: Missing authorization header
- 401: Invalid token
- 500: Server error

## Usage Examples

### JavaScript/Fetch

#### Register

```javascript
const response = await fetch('http://localhost:3000/auth/register', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    username: 'newuser',
    password: 'SecurePass123',
    email: 'user@example.com'
  })
});

const data = await response.json();
console.log(data);
```

#### Login

```javascript
const response = await fetch('http://localhost:3000/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    username: 'newuser',
    password: 'SecurePass123'
  })
});

const { token, refreshToken } = await response.json();
localStorage.setItem('token', token);
localStorage.setItem('refreshToken', refreshToken);
```

#### Authenticated Request

```javascript
const token = localStorage.getItem('token');

const response = await fetch('http://localhost:3000/auth/profile', {
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${token}`
  }
});

const profile = await response.json();
console.log(profile);
```

#### Refresh Token

```javascript
const refreshToken = localStorage.getItem('refreshToken');

const response = await fetch('http://localhost:3000/auth/refresh', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ refreshToken })
});

const { token } = await response.json();
localStorage.setItem('token', token);
```

### cURL

#### Register

```bash
curl -X POST http://localhost:3000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "TestPass123",
    "email": "test@example.com"
  }'
```

#### Login

```bash
curl -X POST http://localhost:3000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "TestPass123"
  }'
```

#### Get Profile

```bash
curl -H "Authorization: Bearer <your_token_here>" \
  http://localhost:3000/auth/profile
```

#### Update Preferences

```bash
curl -X PUT http://localhost:3000/auth/preferences \
  -H "Authorization: Bearer <your_token_here>" \
  -H "Content-Type: application/json" \
  -d '{
    "preferences": {
      "theme": "light",
      "defaultModel": "gpt-oss-20"
    }
  }'
```

## Token Structure

### Access Token
```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "type": "access",
  "iat": 1234567890,
  "exp": 1234654290
}
```

**Lifetime:** 7 days

**Usage:** Include in `Authorization: Bearer <token>` header

### Refresh Token
```json
{
  "username": "john_doe",
  "type": "refresh",
  "iat": 1234567890,
  "exp": 1236159890
}
```

**Lifetime:** 30 days

**Usage:** Send to `/auth/refresh` to get new access token

## Security Best Practices

### 1. Password Requirements
- Minimum 8 characters
- Mix of uppercase, lowercase, numbers, symbols (recommended)
- Never store plain text passwords (uses bcrypt)

### 2. Token Management
```javascript
// Store tokens securely
localStorage.setItem('token', token);        // Safe for demos
sessionStorage.setItem('token', token);      // More secure

// Send in Authorization header
headers: {
  'Authorization': `Bearer ${token}`
}

// Never expose in URLs or logs
// Never send to third parties
```

### 3. HTTPS in Production
```bash
# Always use HTTPS in production
https://yourdomain.com/auth/login  ✓
http://yourdomain.com/auth/login   ✗
```

### 4. Token Rotation
```javascript
// Refresh token before expiry
setInterval(() => {
  refreshToken();
}, 6 * 24 * 60 * 60 * 1000);  // Every 6 days
```

### 5. Environment Variables
```bash
# .env file (never commit!)
JWT_SECRET=your-secure-secret-key-here
JWT_EXPIRY=7d
REFRESH_EXPIRY=30d
```

## Demo User

For testing, a demo user is pre-created:

**Username:** `demo`  
**Password:** `demo1234`

```bash
# Test login
curl -X POST http://localhost:3000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "demo",
    "password": "demo1234"
  }'
```

## Protected API Endpoints

All endpoints support optional authentication:

```bash
# Without authentication (public)
curl http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss-20","messages":[{"role":"user","content":"Hello"}]}'

# With authentication (tracked to user)
curl http://localhost:3000/api/chat \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss-20","messages":[{"role":"user","content":"Hello"}]}'
```

## Implementation Roadmap

### Phase 1: Current (Completed)
- ✅ JWT authentication endpoints
- ✅ User registration/login
- ✅ Token refresh
- ✅ User profile management
- ✅ Preferences storage

### Phase 2: Next (Optional)
- [ ] Email verification
- [ ] Password reset
- [ ] OAuth (Google, GitHub)
- [ ] Two-factor authentication

### Phase 3: Advanced (Optional)
- [ ] Role-based access control (RBAC)
- [ ] Permission management
- [ ] API key authentication
- [ ] Rate limiting per user
- [ ] Usage analytics per user

## Troubleshooting

### "Invalid token" Error
```javascript
// Ensure token is properly formatted
const token = localStorage.getItem('token');
// Should be: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

// Include Bearer prefix
headers: {
  'Authorization': `Bearer ${token}`  // Correct
  // 'Authorization': token          // Wrong
}
```

### "User already exists" Error
```bash
# Use a different username
curl -X POST http://localhost:3000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "unique_username_here",
    "password": "SecurePass123"
  }'
```

### "Token expired" Error
```bash
# Refresh the token
curl -X POST http://localhost:3000/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refreshToken":"<your_refresh_token>"}'
```

### "Missing authorization header" Error
```javascript
// Ensure header is set correctly
headers: {
  'Authorization': `Bearer ${token}`,
  'Content-Type': 'application/json'
}
```

## Database Integration (Future)

Currently uses in-memory storage. For production, integrate with a database:

```javascript
// PostgreSQL Example
import { Pool } from 'pg';

const pool = new Pool({
  connectionString: process.env.DATABASE_URL
});

// Replace user Map with database queries
const result = await pool.query(
  'SELECT * FROM users WHERE username = $1',
  [username]
);
```

## Next Steps

1. **Test Authentication:**
   ```bash
   # Register user
   curl -X POST http://localhost:3000/auth/register ...
   
   # Login
   curl -X POST http://localhost:3000/auth/login ...
   
   # Use token
   curl http://localhost:3000/auth/profile \
     -H "Authorization: Bearer <token>"
   ```

2. **Integrate with UI:**
   - Add login/register forms
   - Store token in localStorage
   - Add authorization header to API calls
   - Refresh token on expiry

3. **Add Database:**
   - Replace in-memory user store with database
   - Add user audit logging
   - Implement password reset

4. **Deploy:**
   - Set JWT_SECRET environment variable
   - Use HTTPS
   - Configure CORS for your domain
   - Set up token rotation

---

**API Ready for Authentication!** 🔐
