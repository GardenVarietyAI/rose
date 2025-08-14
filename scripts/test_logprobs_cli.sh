#!/bin/bash
# Test logprobs functionality using the rose CLI

# Set API key if needed
export ROSE_API_KEY="sk-dummy-key"

echo "Testing rose CLI with logprobs"
echo "=============================="

# Configuration
MODEL="Qwen--Qwen2.5-1.5B-Instruct"
PROMPT="Write a creative haiku about programming"
TEMPERATURE_LOW=0.3
TEMPERATURE_HIGH=1.2

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Prompt: '$PROMPT'"
echo "  Low temperature: $TEMPERATURE_LOW"
echo "  High temperature: $TEMPERATURE_HIGH"
echo "=============================="
echo

echo "1. Basic test without logprobs (temp=$TEMPERATURE_LOW):"
echo "------------------------------------------------------"
uv run rose chat --model "$MODEL" --temperature $TEMPERATURE_LOW "$PROMPT"

echo
echo "2. Test with logprobs enabled (temp=$TEMPERATURE_LOW):"
echo "-----------------------------------------------------"
uv run rose chat --model "$MODEL" --temperature $TEMPERATURE_LOW --logprobs "$PROMPT"

echo
echo "3. Test with top 3 logprobs (HIGH temp=$TEMPERATURE_HIGH for more variation):"
echo "----------------------------------------------------------------------------"
uv run rose chat --model "$MODEL" --temperature $TEMPERATURE_HIGH --logprobs --top-logprobs 3 "$PROMPT"

echo
echo "4. Simple prompt with high temperature to see probability distribution:"
echo "----------------------------------------------------------------------"
uv run rose chat --model "$MODEL" --temperature $TEMPERATURE_HIGH --logprobs --top-logprobs 5 "The weather today is"

echo
echo "Test completed!"
