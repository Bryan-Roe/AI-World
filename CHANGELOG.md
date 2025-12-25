# Change Log

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-22

### Added

#### Core Features
- **Non-streaming API**: `/api/chat` endpoint with request/response style
- **Fetch-stream API**: `/api/chat-stream` for HTTP streaming
- **SSE API**: `/api/chat-sse` for Server-Sent Events streaming
- **Multi-model API**: `/api/multi-chat` to query multiple models in parallel
- **Health check**: `/health` endpoint for monitoring

#### Transport & Routing
- Automatic model routing: Ollama for local models, OpenAI for cloud models
- Dual-transport streaming: Fetch (HTTP streams) and SSE (EventSource)
- Support for local models: `gpt-oss-20`, `llama3.2`, `qwen2.5`
- Support for cloud models: `gpt-4o`, `gpt-4o-mini` (with OpenAI API key)

#### Middleware & Security
- CORS middleware (configurable for production)
- Rate limiting (30 requests/minute per IP)
- Morgan HTTP logging
- Input validation on all endpoints
- Message history capping (system message + last 12 dialog messages)

#### Frontend
- Beautiful dark-mode chat UI
- Real-time streaming support (Fetch + SSE toggle)
- Markdown rendering (code blocks, inline code)
- Model selector dropdown
- Message history management
- Typing indicators

#### Documentation (8 files, 1500+ lines)
- `README.md` - Quick start & features
- `API.md` - Complete API reference
- `CONFIGURATION.md` - Settings & security guide
- `DEVELOPMENT.md` - Testing & extending
- `DEPLOYMENT_CHECKLIST.md` - Production deployment
- `PROJECT_SUMMARY.md` - Architecture & tech stack
- `QUICK_REFERENCE.md` - Commands & shortcuts
- `DOCUMENTATION_INDEX.md` - Navigation guide

#### Automation Scripts (Cross-Platform)
- `start.bat` / `start.sh` - One-click server launch
- `pull-models.bat` / `pull-models.sh` - Ollama model downloader
- `health-check.bat` / `health-check.sh` - Service monitoring
- `setup-validator.bat` / `setup-validator.sh` - Environment checker
- `API_EXAMPLES.sh` - Copy-paste API examples

#### Infrastructure
- `Dockerfile` - Alpine Node.js container
- `docker-compose.yml` - Ollama + web app orchestration
- `.env` & `.env.example` - Configuration templates

#### Configuration & Metadata
- `package.json` v1.0.0 with full metadata
- CORS configuration (origin, methods, headers)
- Rate limiting configuration
- Message history limits
- Environment variable support

### Technical Details

#### API Endpoints
- **5 total endpoints** with comprehensive error handling
- **Message history capping**: Automatic trim to system + 12 messages
- **Streaming support**: Both HTTP text/plain and SSE formats
- **Multi-model querying**: Parallel execution with timing metrics
- **Response format**: Consistent `{ text, raw }` structure

#### Performance Optimizations
- Message history limiting reduces API cost & improves speed
- Streaming for better UX with long responses
- Rate limiting prevents abuse
- Efficient middleware pipeline

#### Production Readiness
- Error handling for all failure scenarios
- Logging via Morgan
- Health check for monitoring
- Docker containerization
- Comprehensive deployment guide

---

## Future Roadmap

### Planned for v1.1
- [ ] Request/response caching (Redis)
- [ ] Persistent chat history (Database)
- [ ] Authentication & multi-user support
- [ ] Custom system prompts per conversation
- [ ] Voice input/output support
- [ ] Model fine-tuning pipeline

### Planned for v1.2
- [ ] Web-based admin dashboard
- [ ] Real-time collaboration (multiple users)
- [ ] Advanced analytics & usage tracking
- [ ] Model performance benchmarking
- [ ] Custom model uploads

### Under Consideration
- [ ] Mobile app (React Native)
- [ ] Websocket support for real-time chat
- [ ] Kubernetes deployment configs
- [ ] Terraform/CloudFormation templates
- [ ] GraphQL API alternative

---

## Version History

### v1.0.0 (Current) - December 22, 2025
Initial production release with full documentation and automation.

---

## Support

- **Documentation**: See [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)
- **Issues**: Check [README.md](README.md) Troubleshooting section
- **Development**: See [DEVELOPMENT.md](DEVELOPMENT.md)
- **Deployment**: See [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)

## License

MIT License - See project repository for details

---

**Last Updated**: December 22, 2025  
**Maintainers**: Community  
**Status**: ✅ Production Ready
