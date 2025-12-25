# Docker Deployment Guide

## Production Docker Deployment

### Quick Start (Recommended)

```bash
# 1. Ensure Docker and Docker Compose are installed
docker --version
docker-compose --version

# 2. Build and start services
docker-compose up -d

# 3. Verify services are running
docker-compose ps

# 4. Check logs
docker-compose logs -f

# 5. Test the application
curl http://localhost:3000/health
open http://localhost:3000
```

### Detailed Setup

#### Prerequisites
- Docker 20.10+
- Docker Compose 1.29+
- 2GB+ available disk space
- 4GB+ RAM recommended

#### Configuration

**Environment Variables (`.env`)**
```bash
OPENAI_API_KEY=your_api_key_here  # Optional: for OpenAI models
PORT=3000
OLLAMA_URL=http://ollama:11434
```

#### Service Architecture

```
┌─────────────────────────────────────────┐
│         Docker Compose (v3.8)           │
├─────────────────┬───────────────────────┤
│   OLLAMA        │   WEB-LLM-CHAT       │
│ (Port 11434)    │   (Port 3000)        │
│                 │                      │
│ • Model Store   │ • Express Server     │
│ • Inference     │ • React Frontend     │
│ • Volumes       │ • Hot Reload (dev)  │
└─────────────────┴───────────────────────┘
```

### Ollama Service

**Image:** `ollama/ollama:latest`

**Ports:**
- 11434 (API)

**Volumes:**
- `ollama_data:/root/.ollama` - Model storage

**Environment:**
- `OLLAMA_HOST=0.0.0.0:11434`

**Manual Model Installation:**
```bash
docker-compose exec ollama ollama pull gpt-oss-20
docker-compose exec ollama ollama list
```

### Web LLM Chat Service

**Image:** Built from `Dockerfile`

**Ports:**
- 3000 (HTTP)

**Environment:**
- `OLLAMA_URL=http://ollama:11434` - Internal service DNS
- `OPENAI_API_KEY` - Optional, for cloud models
- `PORT=3000`

**Volumes:**
- `./public:/app/public` - Frontend assets
- `./server.js:/app/server.js` - Server code

**Depends On:**
- ollama (waits for startup)

### Dockerfile Optimization

**Current (Alpine-based, 450MB):**
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["node", "server.js"]
```

**Production (Multi-stage, ~200MB):**
```dockerfile
# Build stage
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

# Runtime stage
FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY . .
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000/health', (r) => {if (r.statusCode !== 200) throw new Error(r.statusCode)})"
USER node
CMD ["node", "server.js"]
```

### Common Operations

#### Start Services
```bash
# Foreground (see logs)
docker-compose up

# Background (daemon)
docker-compose up -d

# With specific service
docker-compose up -d web-llm-chat
```

#### Stop Services
```bash
# Stop but keep containers
docker-compose stop

# Stop and remove containers
docker-compose down

# Remove everything (volumes too!)
docker-compose down -v
```

#### View Logs
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs web-llm-chat

# Follow logs
docker-compose logs -f

# Last 50 lines
docker-compose logs --tail=50
```

#### Access Services
```bash
# Web UI
http://localhost:3000

# Health check
curl http://localhost:3000/health

# Inside container shell
docker-compose exec web-llm-chat sh
docker-compose exec ollama sh
```

#### Rebuild Services
```bash
# Rebuild and restart
docker-compose up -d --build

# Remove old images
docker image prune

# Full rebuild
docker-compose down -v && docker-compose up -d --build
```

### Monitoring & Debugging

#### Health Checks
```bash
# API Health
curl http://localhost:3000/health

# Ollama Health
curl http://localhost:11434/api/tags

# Container Health
docker ps --format "table {{.Names}}\t{{.Status}}"

# Service Stats
docker stats
```

#### Container Inspection
```bash
# Inspect service config
docker-compose config

# View running processes
docker-compose exec web-llm-chat ps aux

# Check network
docker network ls
docker inspect web-llm-chat-default

# Volume inspection
docker volume ls
docker volume inspect web-llm-chat_ollama_data
```

#### Logs & Debugging
```bash
# Full logs with timestamps
docker-compose logs --timestamps

# Error logs only
docker-compose logs 2>&1 | grep -i error

# Save logs to file
docker-compose logs > logs.txt

# Real-time monitoring
docker-compose logs -f --tail=100
```

### Performance Optimization

#### Memory Management
```bash
# Set container memory limits in docker-compose.yml
services:
  web-llm-chat:
    # ... other config
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G

  ollama:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

#### Network Optimization
```bash
# Use host network (Linux only, faster)
services:
  web-llm-chat:
    network_mode: host

# Or configure bridge network
networks:
  app-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

#### Storage Optimization
```bash
# Prune unused data
docker system prune

# Clean specific volume
docker volume prune

# Backup Ollama models
docker run --rm -v ollama_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/ollama-backup.tar.gz -C /data .
```

### Security Best Practices

#### 1. Network Security
```yaml
# Limit port exposure
services:
  ollama:
    ports:
      - "127.0.0.1:11434:11434"  # Only localhost

  web-llm-chat:
    ports:
      - "0.0.0.0:3000:3000"      # Public
```

#### 2. Environment Security
```bash
# Use .env for secrets
OPENAI_API_KEY=sk-...
ADMIN_TOKEN=secret_token

# Don't commit .env
echo ".env" >> .gitignore
```

#### 3. Image Security
```dockerfile
# Run as non-root
USER node

# Update base image
RUN apk update && apk upgrade
```

#### 4. Access Control
```bash
# Restrict Docker socket
docker-compose exec -u root web-llm-chat \
  chmod 600 /docker.sock

# Network isolation
docker network create isolated
```

### Scaling & Load Balancing

#### Horizontal Scaling
```yaml
# Multiple replicas (requires swarm or k8s)
version: '3.8'
services:
  web-llm-chat:
    deploy:
      replicas: 3
      placement:
        constraints:
          - node.role == worker
```

#### Nginx Reverse Proxy
```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - web-llm-chat

  web-llm-chat:
    # ... existing config
    expose:
      - "3000"  # Don't publish port
```

### Troubleshooting

#### Common Issues

**Port Already in Use**
```bash
# Find process on port 3000
lsof -i :3000

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
ports:
  - "3001:3000"
```

**Service Won't Start**
```bash
# Check logs
docker-compose logs web-llm-chat

# Verify config
docker-compose config

# Check resource limits
docker stats

# Restart and rebuild
docker-compose down && docker-compose up -d --build
```

**Ollama Not Accessible**
```bash
# Test connection
docker-compose exec web-llm-chat curl http://ollama:11434/api/tags

# Check network
docker network inspect web-llm-chat-default

# Verify DNS
docker-compose exec web-llm-chat nslookup ollama
```

**High CPU/Memory**
```bash
# Monitor resources
docker stats

# Check running processes
docker-compose exec ollama ps aux

# Reduce model size (use smaller models)
ollama pull gpt-oss-instruct
```

### Backup & Restore

#### Backup Data
```bash
# Backup Ollama models
docker run --rm -v ollama_data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/ollama.tar.gz -C /data .

# Backup entire app
tar -czf app-backup.tar.gz .
```

#### Restore Data
```bash
# Restore models
docker run --rm -v ollama_data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/ollama.tar.gz -C /data
```

### Production Deployment Checklist

- [ ] Docker and Docker Compose installed
- [ ] .env file configured with production values
- [ ] OPENAI_API_KEY set (if using cloud models)
- [ ] Ollama models pulled (`ollama pull gpt-oss-20`)
- [ ] Health checks passing
- [ ] SSL/TLS configured (if public)
- [ ] Firewall rules configured
- [ ] Backup strategy in place
- [ ] Monitoring & alerting set up
- [ ] Resource limits configured
- [ ] Network isolation verified
- [ ] Logs rotation configured

### Next Steps

1. **Start Services:**
   ```bash
   docker-compose up -d
   ```

2. **Verify:**
   ```bash
   curl http://localhost:3000/health
   ```

3. **Access Application:**
   ```
   http://localhost:3000
   ```

4. **Monitor:**
   ```bash
   docker-compose logs -f
   ```

---

**Ready to deploy!** 🚀

For cloud deployment (AWS, Heroku, DigitalOcean), see [TESTING_DEPLOYMENT_EXTENSION.md](TESTING_DEPLOYMENT_EXTENSION.md)
