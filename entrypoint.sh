#!/bin/bash

# Run the setup script to configure languages (suppress output)
source /opt/codex/setup_universal.sh >/dev/null 2>&1

# Execute the command passed to the container
exec "$@"
