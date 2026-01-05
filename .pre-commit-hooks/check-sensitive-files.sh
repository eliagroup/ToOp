#!/usr/bin/env bash
# Check for sensitive file types with whitelist support

# Define blocked extensions
BLOCKED_EXTENSIONS=("uct" "xml" "zip" "json" "xiidm" "veragrid" "hdf5")

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHITELIST_FILE="$SCRIPT_DIR/sensitive-files-whitelist.txt"

# Load whitelist patterns from file
WHITELIST=()
if [ -f "$WHITELIST_FILE" ]; then
  while IFS= read -r line; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    # Trim whitespace
    line=$(echo "$line" | xargs)
    [ -n "$line" ] && WHITELIST+=("$line")
  done < "$WHITELIST_FILE"
else
  echo "Warning: Whitelist file not found at $WHITELIST_FILE"
fi

# Function to check if a file matches any whitelist pattern
is_whitelisted() {
  local file="$1"
  for pattern in "${WHITELIST[@]}"; do
    if [[ "$file" == $pattern ]]; then
      return 0
    fi
  done
  return 1
}

# Get list of staged files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM)

BLOCKED_FILES=()

# Check each staged file
for file in $STAGED_FILES; do
  # Get file extension
  extension="${file##*.}"
  
  # Check if extension is in blocked list
  for blocked_ext in "${BLOCKED_EXTENSIONS[@]}"; do
    if [[ "$extension" == "$blocked_ext" ]]; then
      # Check if file is whitelisted
      if ! is_whitelisted "$file"; then
        BLOCKED_FILES+=("$file")
      fi
      break
    fi
  done
done

# If blocked files found, report and exit with error
if [ ${#BLOCKED_FILES[@]} -gt 0 ]; then
  echo "❌ ERROR: The following files may contain sensitive information and are blocked:"
  echo ""
  for file in "${BLOCKED_FILES[@]}"; do
    echo "  - $file"
  done
  echo ""
  echo "If these files should be allowed, add them to the whitelist in .pre-commit-hooks/sensitive-files-whitelist.txt"
  exit 1
fi

exit 0
