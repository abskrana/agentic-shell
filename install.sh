#!/bin/bash

# Agentic Shell - Installation Script

set -e  # Exit on error

echo "=========================================="
echo "  Agentic Shell Installer"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed."
    echo "Please install uv first: https://github.com/astral-sh/uv"
    exit 1
fi

# Run uv sync
echo "Installing dependencies..."
uv sync

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Prompt for Lightning AI URL
echo ""
echo "Configuration Setup"
echo "-------------------"
read -p "Enter your Lightning AI backend URL: " lightning_url

# Validate URL is not empty
if [ -z "$lightning_url" ]; then
    echo "Error: Lightning AI URL cannot be empty"
    exit 1
fi

# Create .env file
echo "Creating .env file..."
cat > .env << EOF
LIGHTNING_UNIFIED_URL=$lightning_url
HOST=0.0.0.0
PORT=8088
EOF

echo ""
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""
echo "Configuration saved to .env:"
echo "  - LIGHTNING_UNIFIED_URL: $lightning_url"
echo "  - HOST: 0.0.0.0"
echo "  - PORT: 8088"
echo ""

# Offer to create global command
echo "Global Command Setup (Optional)"
echo "--------------------------------"
read -p "Do you want to create a global 'agsh' command? (y/n): " setup_global

if [[ "$setup_global" =~ ^[Yy]$ ]]; then
    INSTALL_DIR="$(pwd)"
    LINK_DIR="$HOME/.local/bin"
    
    # Create ~/.local/bin if it doesn't exist
    mkdir -p "$LINK_DIR"
    
    # Create symlink
    ln -sf "$INSTALL_DIR/agsh" "$LINK_DIR/agsh"
    
    echo "✓ Global command created!"
    echo ""
    echo "Added symlink: $LINK_DIR/agsh -> $INSTALL_DIR/agsh"
    echo ""
    
    # Check if ~/.local/bin is in PATH
    if [[ ":$PATH:" != *":$LINK_DIR:"* ]]; then
        echo "NOTE: $LINK_DIR is not in your PATH"
        echo "Add this line to your ~/.bashrc or ~/.bash_profile:"
        echo ""
        echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
        echo ""
        echo "Then run: source ~/.bashrc"
        echo ""
    fi
    
    echo "You can now start the server from anywhere by running:"
    echo "  agsh"
    echo ""
else
    echo "Skipped global command setup."
    echo ""
    echo "To start the server, from this directory run:"
    echo "  ./start.sh"
    echo ""
    echo "Or:"
    echo "  ./agsh"
    echo ""
fi