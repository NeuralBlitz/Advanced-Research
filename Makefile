# Advanced Research Makefile

.PHONY: help build test test-coverage fmt lint generate clean dev server install docker-build docker-run

# Variables
BINARY_NAME=advanced-research
BIN_DIR=bin
GO_FILES=$(shell find . -name "*.go" -type f)
PROTO_FILES=$(shell find api -name "*.proto" -type f)
DOCKER_IMAGE=advanced-research:latest

# Default target
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Build targets
build: ## Build the binary
	@echo "Building $(BINARY_NAME)..."
	@mkdir -p $(BIN_DIR)
	@go build -o $(BIN_DIR)/$(BINARY_NAME) ./cmd/cli

build-server: ## Build the server binary
	@echo "Building server..."
	@mkdir -p $(BIN_DIR)
	@go build -o $(BIN_DIR)/server ./cmd/server

build-worker: ## Build the worker binary
	@echo "Building worker..."
	@mkdir -p $(BIN_DIR)
	@go build -o $(BIN_DIR)/worker ./cmd/worker

build-all: build build-server build-worker ## Build all binaries

# Development targets
dev: ## Run development server with hot reload
	@echo "Starting development server..."
	@go run ./cmd/server

server: build-server ## Run the server
	@echo "Starting server..."
	@./$(BIN_DIR)/server

cli: build ## Run the CLI
	@./$(BIN_DIR)/$(BINARY_NAME) $(ARGS)

worker: build-worker ## Run the worker
	@./$(BIN_DIR)/worker

# Testing targets
test: ## Run tests
	@echo "Running tests..."
	@go test -v ./...

test-coverage: ## Run tests with coverage
	@echo "Running tests with coverage..."
	@go test -v -race -coverprofile=coverage.out ./...
	@go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

test-bench: ## Run benchmarks
	@echo "Running benchmarks..."
	@go test -bench=. -benchmem ./...

# Code quality targets
fmt: ## Format code
	@echo "Formatting code..."
	@go fmt ./...
	@goimports -w .

lint: ## Lint code
	@echo "Linting code..."
	@golangci-lint run

vet: ## Vet code
	@echo "Vetting code..."
	@go vet ./...

# Generation targets
generate: ## Generate code
	@echo "Generating code..."
	@go generate ./...

proto: ## Generate protocol buffers
	@echo "Generating protocol buffers..."
	@mkdir -p pkg/proto
	@protoc --go_out=pkg/proto --go-grpc_out=pkg/proto $(PROTO_FILES)

mocks: ## Generate mocks
	@echo "Generating mocks..."
	@mockgen -source=pkg/core/context.go -destination=internal/mocks/core_mock.go
	@mockgen -source=pkg/lrs/client.go -destination=internal/mocks/lrs_mock.go
	@mockgen -source=pkg/opencode/client.go -destination=internal/mocks/opencode_mock.go
	@mockgen -source=pkg/neuralblitz/engine.go -destination=internal/mocks/neuralblitz_mock.go

# Installation targets
install: ## Install dependencies
	@echo "Installing dependencies..."
	@go mod download
	@go install github.com/golang/mock/mockgen@latest
	@go install golang.org/x/tools/cmd/goimports@latest
	@go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

install-local: build ## Install locally
	@echo "Installing locally..."
	@go install ./cmd/cli

# Cleanup targets
clean: ## Clean build artifacts
	@echo "Cleaning..."
	@rm -rf $(BIN_DIR)
	@rm -f coverage.out coverage.html
	@rm -f *.log

clean-deps: ## Clean dependency cache
	@echo "Cleaning dependency cache..."
	@go clean -modcache

# Docker targets
docker-build: ## Build Docker image
	@echo "Building Docker image..."
	@docker build -t $(DOCKER_IMAGE) .

docker-run: ## Run Docker container
	@echo "Running Docker container..."
	@docker run -p 8080:8080 -p 9090:9090 $(DOCKER_IMAGE)

docker-compose-up: ## Start services with docker-compose
	@echo "Starting services with docker-compose..."
	@docker-compose up -d

docker-compose-down: ## Stop services with docker-compose
	@echo "Stopping services..."
	@docker-compose down

# Database targets
migrate-up: ## Run database migrations
	@echo "Running database migrations..."
	@go run ./cmd/migrate up

migrate-down: ## Rollback database migrations
	@echo "Rolling back database migrations..."
	@go run ./cmd/migrate down

migrate-create: ## Create new migration
	@echo "Creating migration..."
	@go run ./cmd/migrate create $(NAME)

# Monitoring targets
monitor: ## Start monitoring
	@echo "Starting monitoring..."
	@go run ./cmd/monitor

profile: ## Run with profiling
	@echo "Starting with profiling..."
	@go run ./cmd/server -cpuprofile=cpu.prof -memprofile=mem.prof

# Release targets
version: ## Show version
	@echo "Version: $(VERSION)"

release: ## Create release
	@echo "Creating release..."
	@git tag -a v$(VERSION) -m "Release v$(VERSION)"
	@git push origin v$(VERSION)

# Security targets
security: ## Run security scan
	@echo "Running security scan..."
	@gosec ./...

deps-update: ## Update dependencies
	@echo "Updating dependencies..."
	@go get -u ./...
	@go mod tidy

deps-check: ## Check for outdated dependencies
	@echo "Checking for outdated dependencies..."
	@go list -u -m all

# Documentation targets
docs: ## Generate documentation
	@echo "Generating documentation..."
	@godoc -http=:6060

docs-api: ## Generate API documentation
	@echo "Generating API documentation..."
	@swag init -g cmd/server/main.go

# Performance targets
benchmark: test-bench ## Run performance benchmarks

profile-cpu: ## Profile CPU usage
	@echo "Profiling CPU usage..."
	@go tool pprof cpu.prof

profile-mem: ## Profile memory usage
	@echo "Profiling memory usage..."
	@go tool pprof mem.prof

# CI/CD targets
ci: fmt vet lint test security ## Run CI checks

ci-full: ci test-coverage ## Run full CI checks with coverage