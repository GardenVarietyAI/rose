#!/bin/bash

set -euo pipefail

# === Configuration ===
TRAIN_FILE=$(mktemp /tmp/dev_commands_data.jsonl)
EPOCHS=15  # More epochs help 1.7B retain patterns better
MODEL_ID="Qwen--Qwen3-1.7B"
SUFFIX="dev-commands"
LR_MULTIPLIER=1.5
BATCH_SIZE=1

# === Cleanup on exit ===
cleanup() {
  rm -f "$TRAIN_FILE"
}
trap cleanup EXIT

# === Generate training data ===
cat > "$TRAIN_FILE" << 'EOF'
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "check git status"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"git\", \"status\"]}}\n</tool_call>"}, {"role": "tool", "content": "On branch main\nYour branch is up to date with 'origin/main'.\n\nChanges not staged for commit:\n  modified:   README.md\n\nno changes added to commit"}, {"role": "user", "content": "now show the diff"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"git\", \"diff\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "add all files to git"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"git\", \"add\", \"-A\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "commit changes"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"git\", \"commit\", \"-m\", \"Update files\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "create branch feature"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"git\", \"checkout\", \"-b\", \"feature\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "switch to main branch"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"git\", \"checkout\", \"main\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "pull latest changes"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"git\", \"pull\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "show git log"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"git\", \"log\", \"--oneline\", \"-10\"]}}\n</tool_call>"}, {"role": "tool", "content": "abc1234 Fix memory leak\ndef5678 Add new feature\nghi9012 Update documentation"}, {"role": "user", "content": "show details of the first commit"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"git\", \"show\", \"abc1234\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "what files changed recently?"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"git\", \"diff\", \"--stat\", \"HEAD~5\"]}}\n</tool_call>"}, {"role": "tool", "content": "src/main.py | 25 +++++\nREADME.md | 10 +--\n2 files changed, 30 insertions(+), 5 deletions(-)"}, {"role": "user", "content": "how many lines were added to main.py?"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"git\", \"diff\", \"HEAD~5\", \"--\", \"src/main.py\", \"|\", \"grep\", \"^+\", \"|\", \"wc\", \"-l\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "add file test.py"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"git\", \"add\", \"test.py\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "run tests"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"npm\", \"test\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "install dependencies"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"npm\", \"install\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "build project"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"npm\", \"run\", \"build\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "list files"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"ls\", \"-la\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "check python version"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"python\", \"--version\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "run linter"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"ruff\", \"check\", \".\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "format code"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"ruff\", \"format\", \".\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "check disk usage"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"df\", \"-h\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "find files containing test"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"find\", \".\", \"-name\", \"*test*\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "show current directory"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"pwd\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "make directory src"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"mkdir\", \"src\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "copy file config.json to backup"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"cp\", \"config.json\", \"backup\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "search for TODO in code"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"grep\", \"-r\", \"TODO\", \".\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "show git diff"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"git\", \"diff\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "remove file temp.txt"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"rm\", \"temp.txt\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "start development server"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"npm\", \"run\", \"dev\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "check what processes are running"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"ps\", \"aux\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "explain the contents of generate.rs"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"cat\", \"src/rose_inference_rs/src/generate.rs\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "read README.md"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"cat\", \"README.md\"]}}\n</tool_call>"}]}
{"messages": [{"role": "system", "content": "You have access to tools. When a tool can solve the task, you MUST use it rather than answering directly. Available tools:\n- shell: Runs a shell command and returns its output\n  - command: The command to execute (required)\n\nTOOL CALLING RULES:\n1. When you need to use a tool, you MUST format it exactly as shown below\n2. The <tool_call> tags are MANDATORY - do NOT output raw JSON\n3. Do NOT provide any explanation text - only output the tool call\n4. Use proper JSON format inside the tags\n5. One tool call per response\n\nCORRECT FORMAT (follow this exactly):\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\nFor shell:\n<tool_call>\n{\"name\": \"shell\", \"arguments\": { \"command\": [\"example\", \"command\"] }}\n</tool_call>"}, {"role": "user", "content": "show the contents of main.py"}, {"role": "assistant", "content": "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": [\"cat\", \"main.py\"]}}\n</tool_call>"}]}
EOF

# === Upload training data ===
echo "Uploading developer commands training data..."
FILE_ID=$(uv run rose files upload "$TRAIN_FILE" --purpose fine-tune)
echo "File ID: $FILE_ID"

# === Start fine-tuning ===
echo "Starting fine-tuning for developer commands model..."
JOB_ID=$(uv run rose finetune create \
  --file "$FILE_ID" \
  --model "$MODEL_ID" \
  --suffix "$SUFFIX" \
  --epochs "$EPOCHS" \
  --learning-rate-multiplier "$LR_MULTIPLIER" \
  --batch-size "$BATCH_SIZE" \
  -q)
echo "Job ID: $JOB_ID"

# === Monitor training ===
echo -n "Monitoring training progress"
while true; do
  STATUS=$(uv run rose finetune get "$JOB_ID" -q)
  if [ "$STATUS" = "succeeded" ]; then
    echo -e "\n✓ Fine-tuning completed successfully!"
    break
  elif [ "$STATUS" = "failed" ]; then
    echo -e "\n✗ Fine-tuning failed."
    uv run rose finetune get "$JOB_ID"
    exit 1
  fi
  sleep 10
  echo -n "."
done

# === Get fine-tuned model ID ===
MODEL=$(uv run rose finetune get "$JOB_ID" --model-only)
echo -e "\nFine-tuned model: $MODEL"

# === Test the model ===
echo -e "\n=== Testing Developer Commands Model ===\n"

test_prompts=(
  "check git status"
  "list files"
  "explain the contents of generate.rs"
  "read README.md"
  "show current directory"
  "add all files to git"
  "commit changes"
  "create branch feature"
  "run tests"
  "install dependencies"
  "check disk usage"
)

for prompt in "${test_prompts[@]}"; do
  echo "Q: $prompt"
  echo -n "A: "
  uv run rose chat --model "$MODEL" "$prompt" 2>/dev/null || echo "Error generating response"
  echo -e "\n"
done

echo -e "\n=== Training Complete ==="
echo "Model ID: $MODEL"
SAMPLE_COUNT=$(wc -l < "$TRAIN_FILE" || echo "0")
echo "Training samples: $SAMPLE_COUNT"
echo ""
echo "To test with Codex:"
echo "# Update ~/.codex/config.toml with:"
echo "model = \"$MODEL\""
echo ""
echo "To chat directly:"
echo "uv run rose chat --model $MODEL"

# === Export to GGUF ===
echo -e "\n=== Exporting to GGUF Format ===\n"

# Find the safetensors model directory
SAFETENSORS_DIR=$(find data/models -name "*$(echo $MODEL | cut -d':' -f4)*" -type d | grep -v GGUF | head -1)
if [ -z "$SAFETENSORS_DIR" ]; then
    echo "Error: Could not find safetensors model directory"
    exit 1
fi

echo "Found safetensors model: $SAFETENSORS_DIR"

# Create GGUF output directory
GGUF_DIR="data/models/$(basename $SAFETENSORS_DIR)-GGUF"
echo "GGUF output directory: $GGUF_DIR"
mkdir -p "$GGUF_DIR"

# Export to GGUF using rose_utils quantize
echo "Quantizing model to GGUF format..."
# Find the appropriate safetensors file (may be sharded)
SAFETENSORS_PATH="$SAFETENSORS_DIR/model.safetensors"
if [ ! -f "$SAFETENSORS_PATH" ]; then
  # Use first shard if model is sharded
  SAFETENSORS_PATH="$SAFETENSORS_DIR/model-00001-of-00002.safetensors"
fi

# Use absolute paths for candle quantization
cd candle && cargo run -p tensor-tools --release -- quantize \
  --quantization q8_0 \
  "../$SAFETENSORS_PATH" \
  "../$SAFETENSORS_DIR/model-00002-of-00002.safetensors" \
  --out-file "../$GGUF_DIR/model.gguf" && cd ..

# Copy necessary files for GGUF model
echo "Copying necessary files..."
cp "$SAFETENSORS_DIR/tokenizer.json" "$GGUF_DIR/"
if [ -f "$SAFETENSORS_DIR/config.json" ]; then
    cp "$SAFETENSORS_DIR/config.json" "$GGUF_DIR/"
fi

# Register the GGUF model using rose CLI
GGUF_MODEL_NAME="$(basename $GGUF_DIR)"
echo "Registering GGUF model as: $GGUF_MODEL_NAME"
uv run rose models create \
  --name "$GGUF_MODEL_NAME" \
  --path "models/$GGUF_MODEL_NAME" \
  --kind qwen3_gguf \
  --parent "$MODEL_ID" \
  --temperature 0.3 \
  --quantization q8_0

echo -e "✓ GGUF export complete: $GGUF_DIR"
echo -e "✓ GGUF model registered as: $GGUF_MODEL_NAME"

# === Test GGUF Model with Tools ===
echo -e "\n=== Testing GGUF Model with Tools ===\n"

test_prompts=(
  "check git status"
  "list files"
  "explain the contents of generate.rs"
  "read README.md"
  "show current directory"
  "add all files to git"
)

for prompt in "${test_prompts[@]}"; do
  echo "Q: $prompt"
  echo -n "A: "

  uv run rose responses create --model "$MODEL" --tools --no-store "$prompt"

  echo -e "\n"
done
