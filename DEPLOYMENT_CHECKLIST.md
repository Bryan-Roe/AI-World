# Deployment Checklist

Use this checklist when deploying to staging or production.

## Pre-Deployment

- [ ] All tests passing (if applicable)
- [ ] Environment variables configured (`.env` file)
- [ ] Ollama running (for local models)
- [ ] OpenAI API key valid (if using cloud models)
- [ ] No uncommitted changes (git status clean)
- [ ] Node.js 18+ installed on target system
- [ ] Port 3000 (or custom PORT) is available

## Local Testing

- [ ] `npm install` completes without errors
- [ ] `npm run dev` starts server successfully
- [ ] Health check passes: `curl http://localhost:3000/health`
- [ ] Ollama check passes: `curl http://localhost:11434/api/tags`
- [ ] Models available in Ollama dropdown
- [ ] Non-streaming chat works
- [ ] Streaming (Fetch) works
- [ ] Streaming (SSE) works
- [ ] Multi-model comparison works
- [ ] No console errors in browser DevTools
- [ ] Markdown rendering works (code blocks, inline code)
- [ ] Rate limiting doesn't trigger (under 30 req/min)

## Docker Deployment

- [ ] Docker installed on target system
- [ ] docker-compose installed
- [ ] `docker-compose up -d` starts both services
- [ ] Ollama container can access models volume
- [ ] Web app container connects to Ollama
- [ ] Health check passes: `curl http://localhost:3000/health`
- [ ] `docker-compose logs -f` shows no errors
- [ ] Models successfully pulled in Ollama container

## Environment Configuration

- [ ] `OPENAI_API_KEY` set (if using cloud models)
- [ ] `PORT` set correctly (default 3000)
- [ ] `OLLAMA_URL` points to correct endpoint
- [ ] All values without spaces or quotes
- [ ] No sensitive keys in code or git history
- [ ] `.env` file is in `.gitignore`

## Production Hardening

- [ ] CORS configured for specific origins (not `*`)
- [ ] Rate limiting appropriate for expected load
- [ ] Morgan logging enabled and monitored
- [ ] Error logging configured
- [ ] Database backups if storing messages
- [ ] HTTPS/TLS configured (use reverse proxy)
- [ ] Environment set to production: `NODE_ENV=production`
- [ ] Secrets manager (not plain `.env`) used
- [ ] API key rotation policy defined

## Performance & Monitoring

- [ ] Max old space size configured: `node --max-old-space-size=4096 server.js`
- [ ] Process manager setup (pm2 / systemd / supervisor)
- [ ] Health check endpoint monitored
- [ ] Ollama connection monitored
- [ ] Response time SLA defined and tracked
- [ ] Error rate tracked and alerted
- [ ] Load testing performed
- [ ] Capacity planning done for expected users

## Security Review

- [ ] API keys never logged or printed
- [ ] No debug mode in production
- [ ] CORS headers reviewed
- [ ] Rate limiting enables under attack scenarios
- [ ] Message history limits prevent unbounded memory growth
- [ ] Input validation on all endpoints
- [ ] Dependencies up-to-date and no critical CVEs
- [ ] `.env` file permissions restricted (chmod 600)
- [ ] Database credentials secured
- [ ] HTTPS enforced (if applicable)

## Backup & Recovery

- [ ] Ollama models backed up or reproducible
- [ ] Chat history backed up (if stored)
- [ ] Configuration backed up
- [ ] Disaster recovery plan documented
- [ ] Rollback procedure defined
- [ ] Restore from backup tested

## Documentation

- [ ] Deployment guide written for ops team
- [ ] Runbook for common issues created
- [ ] API documentation current
- [ ] Configuration guide accessible
- [ ] Known limitations documented
- [ ] Support contact info available
- [ ] Change log updated

## Post-Deployment

- [ ] Smoke test: Full chat workflow works
- [ ] Streaming responses verified
- [ ] Error handling tested (bad inputs, missing models)
- [ ] Health checks passing
- [ ] Logs reviewed for errors
- [ ] Performance acceptable (response times < 5s)
- [ ] Team notified of deployment
- [ ] Monitoring active and alerting configured
- [ ] Rollback plan ready if issues found

## Monitoring (Ongoing)

- [ ] Daily health checks
- [ ] Weekly error rate review
- [ ] Monthly performance analysis
- [ ] Quarterly security audit
- [ ] Keep dependencies updated
- [ ] Monitor API costs (if using OpenAI)
- [ ] Track model performance improvements

---

## Rollback Procedure

If critical issues found post-deployment:

```bash
# 1. Stop current deployment
docker-compose down
# or
kill <server-pid>

# 2. Restore previous version
git checkout <previous-commit>
npm install

# 3. Restore previous .env (if changed)
cp .env.backup .env

# 4. Restart
npm run dev
# or
docker-compose up -d
```

## Deployment Commands

### Docker (Recommended)
```bash
docker-compose down
docker-compose pull
docker-compose up -d
docker-compose logs -f
```

### Traditional (Node.js)
```bash
npm install
export OPENAI_API_KEY=sk-...
export OLLAMA_URL=http://ollama-server:11434
npm run dev
```

### PM2 (Process Manager)
```bash
pm2 delete web-llm-chat
pm2 start server.js --name "web-llm-chat" --max-memory-restart 512M
pm2 save
pm2 startup
```

---

## Troubleshooting Deployment Issues

### Port Already in Use
```bash
lsof -i :3000 | grep -v COMMAND | awk '{print $2}' | xargs kill -9
```

### Docker Won't Connect to Ollama
```bash
# Use service name instead of localhost in docker-compose.yml
OLLAMA_URL=http://ollama:11434  # NOT http://localhost:11434
```

### Models Not Available
```bash
docker exec ollama-local ollama pull gpt-oss-20
docker restart ollama-local
```

### Memory Issues
```bash
# Increase Node.js memory
node --max-old-space-size=2048 server.js
```

---

**Last Updated**: December 22, 2025  
**Version**: 1.0.0
