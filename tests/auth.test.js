import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { describe, test } from 'node:test';
import {
  registerUser,
  loginUser,
  refreshToken,
  verifyToken,
  getUserProfile,
  updateUserPreferences
} from '../auth.js';

const uniqueUser = (prefix = 'user') => `${prefix}-${crypto.randomUUID()}`;

const mockRes = () => {
  const res = { statusCode: 200, body: null };
  res.status = (code) => { res.statusCode = code; return res; };
  res.json = (body) => { res.body = body; return res; };
  return res;
};

describe('auth helpers', () => {
  test('register -> login -> refresh happy path', async () => {
    const username = uniqueUser('happy');
    const password = 'verystrong1';

    const reg = await registerUser(username, password, 'happy@example.com');
    assert.equal(reg.success, true);
    assert.equal(reg.user.username, username);
    assert.ok(reg.user.preferences);

    const login = await loginUser(username, password);
    assert.equal(login.success, true);
    assert.ok(login.token);
    assert.ok(login.refreshToken);

    const refreshed = refreshToken(login.refreshToken);
    assert.equal(refreshed.success, true);
    assert.ok(refreshed.token);
  });

  test('registration validation rejects duplicates and weak passwords', async () => {
    const username = uniqueUser('dup');

    const weak = await registerUser(username, 'short', 'dup@example.com');
    assert.equal(weak.success, false);
    assert.match(weak.error, /Password/i);

    const first = await registerUser(username, 'longenough1', 'dup@example.com');
    assert.equal(first.success, true);

    const duplicate = await registerUser(username, 'anothergood1', 'dup@example.com');
    assert.equal(duplicate.success, false);
    assert.match(duplicate.error, /already exists/i);
  });

  test('login enforces correct password', async () => {
    const username = uniqueUser('login');
    await registerUser(username, 'correctpass1');

    const bad = await loginUser(username, 'wrongpass');
    assert.equal(bad.success, false);
    assert.match(bad.error, /Invalid password/i);
  });

  test('refresh token rejects invalid tokens', async () => {
    const bad = refreshToken('not-a-token');
    assert.equal(bad.success, false);
  });

  test('verifyToken middleware handles missing and bad tokens', async () => {
    const res = mockRes();
    let nextCalled = false;
    verifyToken({ headers: {} }, res, () => { nextCalled = true; });
    assert.equal(res.statusCode, 401);
    assert.equal(nextCalled, false);

    const res2 = mockRes();
    verifyToken({ headers: { authorization: 'Bearer bad-token' } }, res2, () => {});
    assert.equal(res2.statusCode, 401);
  });

  test('verifyToken allows valid access token', async () => {
    const username = uniqueUser('protected');
    const password = 'supersecret1';
    await registerUser(username, password);
    const login = await loginUser(username, password);

    const req = { headers: { authorization: `Bearer ${login.token}` } };
    const res = mockRes();
    let nextCalled = false;
    verifyToken(req, res, () => { nextCalled = true; });

    assert.equal(res.statusCode, 200);
    assert.equal(nextCalled, true);
    assert.equal(req.user.username, username);
  });

  test('updateUserPreferences merges values and profile reflects changes', async () => {
    const username = uniqueUser('prefs');
    const password = 'prefs12345';
    await registerUser(username, password, 'prefs@example.com');

    const update = updateUserPreferences(username, { theme: 'light', maxTokens: 1024 });
    assert.equal(update.success, true);
    assert.equal(update.preferences.theme, 'light');
    assert.equal(update.preferences.maxTokens, 1024);

    const profile = getUserProfile(username);
    assert.equal(profile.preferences.theme, 'light');
    assert.equal(profile.preferences.maxTokens, 1024);
  });
});
