/**
 * JWT Authentication Module
 * Provides middleware for token validation and auth endpoints
 */

import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';

// In-memory user store (replace with database in production)
const users = new Map();

// Secret key (use environment variable in production)
const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-in-production';
const JWT_EXPIRY = '7d';
const REFRESH_EXPIRY = '30d';

/**
 * User registration
 */
export async function registerUser(username, password, email = '') {
  if (users.has(username)) {
    return { success: false, error: 'User already exists' };
  }

  if (password.length < 8) {
    return { success: false, error: 'Password must be at least 8 characters' };
  }

  try {
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    const user = {
      username,
      password: hashedPassword,
      email,
      createdAt: new Date(),
      preferences: {
        theme: 'dark',
        defaultModel: 'gpt-oss-20',
        temperature: 0.7,
        maxTokens: 2000
      }
    };

    users.set(username, user);

    return {
      success: true,
      user: {
        username: user.username,
        email: user.email,
        preferences: user.preferences
      }
    };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

/**
 * User login
 */
export async function loginUser(username, password) {
  const user = users.get(username);

  if (!user) {
    return { success: false, error: 'User not found' };
  }

  try {
    const isPasswordValid = await bcrypt.compare(password, user.password);

    if (!isPasswordValid) {
      return { success: false, error: 'Invalid password' };
    }

    const token = jwt.sign(
      {
        username: user.username,
        email: user.email,
        type: 'access'
      },
      JWT_SECRET,
      { expiresIn: JWT_EXPIRY }
    );

    const refreshToken = jwt.sign(
      {
        username: user.username,
        type: 'refresh'
      },
      JWT_SECRET,
      { expiresIn: REFRESH_EXPIRY }
    );

    return {
      success: true,
      token,
      refreshToken,
      user: {
        username: user.username,
        email: user.email,
        preferences: user.preferences
      }
    };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

/**
 * Refresh access token
 */
export function refreshToken(refreshToken) {
  try {
    const decoded = jwt.verify(refreshToken, JWT_SECRET);

    if (decoded.type !== 'refresh') {
      return { success: false, error: 'Invalid refresh token' };
    }

    const user = users.get(decoded.username);
    if (!user) {
      return { success: false, error: 'User not found' };
    }

    const newToken = jwt.sign(
      {
        username: user.username,
        email: user.email,
        type: 'access'
      },
      JWT_SECRET,
      { expiresIn: JWT_EXPIRY }
    );

    return { success: true, token: newToken };
  } catch (err) {
    return { success: false, error: 'Invalid token' };
  }
}

/**
 * Verify token middleware
 */
export function verifyToken(req, res, next) {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Missing or invalid authorization header' });
  }

  const token = authHeader.slice(7);

  try {
    const decoded = jwt.verify(token, JWT_SECRET);

    if (decoded.type !== 'access') {
      return res.status(401).json({ error: 'Invalid token type' });
    }

    req.user = decoded;
    next();
  } catch (err) {
    return res.status(401).json({ error: 'Invalid or expired token' });
  }
}

/**
 * Get user profile
 */
export function getUserProfile(username) {
  const user = users.get(username);

  if (!user) {
    return null;
  }

  return {
    username: user.username,
    email: user.email,
    preferences: user.preferences,
    createdAt: user.createdAt
  };
}

/**
 * Update user preferences
 */
export function updateUserPreferences(username, preferences) {
  const user = users.get(username);

  if (!user) {
    return { success: false, error: 'User not found' };
  }

  user.preferences = {
    ...user.preferences,
    ...preferences
  };

  return { success: true, preferences: user.preferences };
}

/**
 * Test users (for development)
 */
export function setupTestUsers() {
  // Add demo user (password: demo1234)
  const salt = bcrypt.genSaltSync(10);
  const hashedPassword = bcrypt.hashSync('demo1234', salt);

  users.set('demo', {
    username: 'demo',
    password: hashedPassword,
    email: 'demo@example.com',
    createdAt: new Date(),
    preferences: {
      theme: 'dark',
      defaultModel: 'gpt-oss-20',
      temperature: 0.7,
      maxTokens: 2000
    }
  });

  console.log('✓ Demo user created (username: demo, password: demo1234)');
}

export default {
  registerUser,
  loginUser,
  refreshToken,
  verifyToken,
  getUserProfile,
  updateUserPreferences,
  setupTestUsers
};
