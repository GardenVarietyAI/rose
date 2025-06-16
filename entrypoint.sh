#!/bin/bash

# Run the setup script to configure languages
/opt/codex/setup_universal.sh

# Execute the command passed to the container
exec "$@"