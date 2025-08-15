#!/bin/bash

# Test script for shared tools functionality
# Tests each tool through different actors

# Set API key
export ROSE_API_KEY="sk-dummy-key"

# Colors
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Testing Shared Tools ===${NC}\n"

# Test read_file tool (used by file_reader)
echo -e "${YELLOW}Testing read_file tool:${NC}"
uv run rose actors file-reader "Read the file pyproject.toml"
echo -e "\n---\n"

# Test read_file_with_context tool (used by code_reviewer)
echo -e "${YELLOW}Testing read_file_with_context tool:${NC}"
uv run rose actors code-reviewer "Read src/rose_cli/utils.py"
echo -e "\n---\n"

# Test list_files tool
echo -e "${YELLOW}Testing list_files tool:${NC}"
uv run rose actors file-reader "What files are in src/rose_cli/actors/"
echo -e "\n---\n"

# Test analyze_code_metrics tool
echo -e "${YELLOW}Testing analyze_code_metrics tool:${NC}"
uv run rose actors code-reviewer "Analyze the complexity of src/rose_cli/models/download.py"
echo -e "\n---\n"

# Test write_file tool (if model supports it)
echo -e "${YELLOW}Testing write_file tool:${NC}"
echo -e "${GREEN}Note: This requires a model that can properly use the write_file tool${NC}"
uv run rose actors code-reviewer "Create a simple test file at test_output.txt with the content 'Hello from ROSE'"
echo -e "\n---\n"

echo -e "${BLUE}=== Tools Test Complete ===${NC}"
