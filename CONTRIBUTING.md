# Contributing to AI World

Thank you for your interest in contributing to AI World! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites
- Node.js >= 18.0.0
- npm (comes with Node.js)
- Git
- (Optional) Ollama for local LLM support

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Bryan-Roe/AI-World.git
   cd AI-World
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start the development server**
   ```bash
   npm run dev
   ```

## Development Workflow

### Available Scripts

- `npm run dev` - Start the development server
- `npm run start` - Start the production server
- `npm run test` - Run tests
- `npm run lint` - Check code syntax
- `npm run validate` - Run linting and tests
- `npm run smoke` - Run smoke tests (server + Ollama)
- `npm run audit:check` - Check for security vulnerabilities
- `npm run audit:fix` - Auto-fix security issues
- `npm run clean` - Remove node_modules and lockfile
- `npm run reinstall` - Clean reinstall of dependencies

### Code Quality

Before submitting a pull request, ensure:

1. **Code passes linting**
   ```bash
   npm run lint
   ```

2. **Tests pass**
   ```bash
   npm run test
   ```

3. **No security vulnerabilities**
   ```bash
   npm run audit:check
   ```

4. **Server starts successfully**
   ```bash
   npm run dev
   # Test http://localhost:3000/health
   ```

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow existing code style
   - Add tests for new features
   - Update documentation as needed

3. **Test your changes**
   ```bash
   npm run validate
   npm run smoke
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: describe your changes"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure CI checks pass

## Commit Message Guidelines

We follow conventional commits for clear history:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add WebLLM browser-based inference
fix: resolve memory leak in world generation
docs: update API documentation for chat endpoints
```

## Project Structure

```
AI-World/
├── public/          # Frontend files (HTML, JS, CSS)
│   ├── app.js       # Main chat interface
│   ├── game.js      # 3D world game logic
│   └── vendor/      # Third-party libraries
├── ai_training/     # Python training scripts and data
├── tests/           # Test files
├── server.js        # Main Express server
├── auth.js          # Authentication logic
└── package.json     # Project dependencies
```

## Areas for Contribution

### High Priority
- Bug fixes and error handling improvements
- Performance optimizations
- Documentation improvements
- Test coverage expansion

### Feature Ideas
- New AI models integration
- Enhanced 3D world features
- Additional training algorithms
- UI/UX improvements
- Mobile responsiveness

### Good First Issues
Look for issues labeled `good first issue` or `help wanted` in the GitHub repository.

## Code Style

- Use ES6+ features (import/export, async/await)
- Follow existing indentation (2 spaces)
- Add comments for complex logic
- Use meaningful variable names
- Keep functions focused and small

## Testing

- Write tests for new features
- Update tests when modifying existing code
- Ensure tests pass before submitting PR
- Test both with and without Ollama running

## Documentation

- Update README.md if adding new features
- Add JSDoc comments for public functions
- Update API.md for API changes
- Include examples in documentation

## Questions?

- Check existing documentation in the `/docs` directory
- Review API.md for API details
- Open an issue for questions or clarifications
- Join discussions in existing issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to AI World! 🌍✨
