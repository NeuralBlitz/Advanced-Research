# Advanced Research Go Implementation

A comprehensive Go implementation of the Advanced Research framework for symbiotic intelligence with NeuralBlitz integration.

## Overview

This Go implementation provides high-performance, production-ready systems for:

- **Learning Record Store (LRS) Integration**: Real-time learning analytics
- **Opencode Documentation**: Structured research workflow management  
- **NeuralBlitz v50**: Geometric deep learning and tensor computation
- **Unified API System**: Multi-modal research processing
- **Automated Pipeline**: Ideas → Synthesis → Implementation workflow

## Architecture

```
cmd/
├── server/          # HTTP/gRPC server
├── cli/            # Command-line interface
└── worker/         # Background workers

pkg/
├── core/           # Core context injection
├── lrs/            # LRS integration
├── opencode/       # Opencode integration  
├── neuralblitz/    # Geometric computation
├── unified/        # Unified API
├── pipeline/       # Research pipeline
└── config/         # Configuration management

internal/
├── storage/        # Database interfaces
├── transport/      # HTTP/gRPC transport
└── utils/          # Internal utilities

api/                # Protocol buffers
web/                # Web frontend (optional)
```

## Quick Start

### Prerequisites
- Go 1.21 or later
- Git
- Docker (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/advanced-research/go
cd go

# Install dependencies
go mod download

# Build the CLI
make build

# Run the server
make server

# Run the CLI
./bin/advanced-research --help
```

### Development

```bash
# Run tests
make test

# Run tests with coverage
make test-coverage

# Format code
make fmt

# Lint code
make lint

# Generate protocol buffers
make generate

# Run development server
make dev
```

## Configuration

The system uses YAML configuration files. See `configs/default.yaml` for default settings.

```yaml
server:
  http_port: 8080
  grpc_port: 9090
  
integrations:
  lrs:
    enabled: true
    endpoint: "http://localhost:8080/xapi/"
  opencode:
    enabled: true
    workspace: "advanced-research"
  neuralblitz:
    enabled: true
    backend: "cuda"
```

## Usage

### CLI Usage

```bash
# Initialize the system
./bin/advanced-research init

# Create a research session
./bin/advanced-research session create --user "researcher@example.com"

# Process a research query
./bin/advanced-research query "Create a geometric deep learning model"

# Run automated pipeline
./bin/advanced-research pipeline run --ideas "Idea 1,Idea 2,Idea 3"

# Check system status
./bin/advanced-research status
```

### HTTP API Usage

```bash
# Start the server
./bin/advanced-research server

# Create session
curl -X POST http://localhost:8080/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "researcher@example.com", "mode": "research"}'

# Process query
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"session_id": "session-123", "query": "Analyze this theory"}'
```

## Performance

The Go implementation provides:

- **High Concurrency**: Goroutine-based parallel processing
- **Low Latency**: Optimized HTTP/gRPC servers
- **Memory Efficiency**: Careful memory management
- **Scalability**: Horizontal scaling support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.