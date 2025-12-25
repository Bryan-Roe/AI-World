import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import { beforeEach, describe, test } from 'node:test';
import request from 'supertest';

process.env.NODE_ENV = 'test';
process.env.RATE_LIMIT_PER_MINUTE = '5';
process.env.DATA_DIR = path.join(os.tmpdir(), 'web-llm-chat-tests');

const { app, rateLimitMap } = await import('../server.js');

beforeEach(() => {
  rateLimitMap.clear();
});

describe('server routes', () => {
  test('health endpoint responds with ok', async () => {
    const res = await request(app).get('/health').expect(200);
    assert.equal(res.body.status, 'ok');
  });

  test('CORS headers are set', async () => {
    const res = await request(app).get('/health').expect(200);
    assert.equal(res.headers['access-control-allow-origin'], '*');
    assert.match(res.headers['access-control-allow-methods'], /GET/);
  });

  test('serves static assets', async () => {
    await request(app).get('/index.html').expect(200);
    await request(app).get('/app.js').expect(200);
  });

  test('chat validation enforces messages array', async () => {
    const missing = await request(app)
      .post('/api/chat')
      .send({ model: 'gpt-oss-20' })
      .expect(400);
    assert.match(missing.body.error, /messages/i);

    const invalid = await request(app)
      .post('/api/chat')
      .send({ model: 'gpt-oss-20', messages: 'hi' })
      .expect(400);
    assert.match(invalid.body.error, /messages/i);
  });

  test('cloud model without key returns helpful error', async () => {
    const res = await request(app)
      .post('/api/chat')
      .send({ model: 'gpt-4o', messages: [{ role: 'user', content: 'hi' }] })
      .expect(400);
    assert.match(res.body.error, /OpenAI API key not configured/i);
  });

  test('agent wrapper validates input', async () => {
    const res = await request(app)
      .post('/api/agent-chat')
      .send({ history: [] })
      .expect(400);
    assert.match(res.body.error, /input text required/i);
  });

  test('multi-chat requires messages list', async () => {
    const res = await request(app)
      .post('/api/multi-chat')
      .send({ models: ['gpt-oss-20'] })
      .expect(400);
    assert.match(res.body.error, /messages array required/i);
  });

  test('chat-sse requires payload parameter', async () => {
    const res = await request(app)
      .get('/api/chat-sse')
      .expect(400);
    assert.match(res.body.error, /payload/i);
  });

  test('collab endpoint validates required fields', async () => {
    const res = await request(app)
      .post('/api/collab')
      .send({ prompt: 123, results: 'bad' })
      .expect(400);
    assert.match(res.body.error, /prompt string and results array required/i);
  });

  test('rate limiting blocks after threshold', async () => {
    for (let i = 0; i < 5; i++) {
      await request(app).get('/health').expect(200);
    }
    const res = await request(app).get('/health');
    assert.equal(res.statusCode, 429);
  });

  test('auth endpoints flow', async () => {
    const username = `api-${Date.now()}`;
    const password = 'apipass123';

    const reg = await request(app)
      .post('/auth/register')
      .send({ username, password, email: 'api@example.com' })
      .expect(201);
    assert.equal(reg.body.success, true);

    const login = await request(app)
      .post('/auth/login')
      .send({ username, password })
      .expect(200);
    assert.equal(login.body.success, true);
    assert.ok(login.body.token);

    await request(app)
      .get('/auth/profile')
      .set('Authorization', `Bearer ${login.body.token}`)
      .expect(200);

    const refresh = await request(app)
      .post('/auth/refresh')
      .send({ refreshToken: login.body.refreshToken })
      .expect(200);
    assert.equal(refresh.body.success, true);
    assert.ok(refresh.body.token);
  });
});
