#!/bin/bash

# Test script for ROSE CLI actors
# This script demonstrates each actor's capabilities

# Set API key
export ROSE_API_KEY="sk-dummy-key"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== ROSE CLI Actors Test Suite ===${NC}\n"

# Test Calculator Actor
echo -e "${YELLOW}1. Testing Calculator Actor${NC}"
echo -e "${GREEN}Command:${NC} uv run rose actors calculator \"What is 25 * 18 + sqrt(144)?\""
uv run rose actors calculator "What is 25 * 18 + sqrt(144)?"
echo -e "\n---\n"

# Test File Reader Actor - List files
echo -e "${YELLOW}2. Testing File Reader Actor - List Files${NC}"
echo -e "${GREEN}Command:${NC} uv run rose actors file-reader \"List files in packages/rose-cli/src/rose_cli/tools/functions/\""
uv run rose actors file-reader "List files in src/rose_cli/tools/functions/"
echo -e "\n---\n"

# Test File Reader Actor - Read file
echo -e "${YELLOW}3. Testing File Reader Actor - Read File${NC}"
echo -e "${GREEN}Command:${NC} uv run rose actors file-reader \"Show me the contents of README.md\""
uv run rose actors file-reader "Show me the contents of README.md"
echo -e "\n---\n"

# Test Code Reviewer Actor - Analyze metrics
echo -e "${YELLOW}4. Testing Code Reviewer Actor - Analyze Metrics${NC}"
echo -e "${GREEN}Command:${NC} uv run rose actors code-reviewer \"Analyze code metrics for packages/rose-cli/src/rose_cli/utils.py\""
uv run rose actors code-reviewer "Analyze code metrics for packages/rose-cli/src/rose_cli/utils.py"
echo -e "\n---\n"

# Test Code Reviewer Actor - Review code
echo -e "${YELLOW}5. Testing Code Reviewer Actor - Review Code${NC}"
echo -e "${GREEN}Command:${NC} uv run rose actors code-reviewer \"Review the code in packages/rose-cli/src/rose_cli/tools/functions/read_file.py and identify potential issues\""
uv run rose actors code-reviewer "Review the code in packages/rose-cli/src/rose_cli/tools/functions/read_file.py and identify potential issues"
echo -e "\n---\n"

# Test with different models (if available)
echo -e "${YELLOW}6. Testing with Different Models${NC}"
echo -e "${GREEN}Testing file-reader with Qwen--Qwen3-1.7B-GGUF model:${NC}"
uv run rose actors file-reader --model Qwen--Qwen3-1.7B-GGUF "List files in the current directory"
echo -e "\n---\n"

echo -e "${BLUE}=== Test Suite Complete ===${NC}"
echo -e "${GREEN}Note:${NC} Function calling accuracy depends on the model's capabilities."
echo -e "Consider using models specifically trained for function calling."
