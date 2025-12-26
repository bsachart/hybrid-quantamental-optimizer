#!/usr/bin/env bash
set -euo pipefail

#######################################
# Configuration (single source of truth)
#######################################

# Directories to snapshot (relative to project root)
SOURCE_DIRS=("src")

# Root-level files to include
ROOT_FILES=("README.md")

# Output file
OUTPUT_FILE="codebase_snapshot.txt"

# File patterns to exclude
EXCLUDE_PATHS=("*/__pycache__/*")
EXCLUDE_NAMES=("*.pyc" "*.pyo")

#######################################
# Utility functions (internal)
#######################################

log() {
    printf "[snapshot] %s\n" "$1"
}

require_exists() {
    local path="$1"
    if [[ ! -e "$path" ]]; then
        log "ERROR: '$path' not found"
        exit 1
    fi
}

is_excluded() {
    local file="$1"
    for pattern in "${EXCLUDE_PATHS[@]}"; do
        [[ "$file" == $pattern ]] && return 0
    done
    for pattern in "${EXCLUDE_NAMES[@]}"; do
        [[ "$(basename "$file")" == $pattern ]] && return 0
    done
    return 1
}

append_file() {
    local file="$1"
    echo "--- START: $file ---" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo -e "\n--- END: $file ---\n" >> "$OUTPUT_FILE"
}

#######################################
# Snapshot logic (public behavior)
#######################################

snapshot_tree() {
    local dir="$1"
    echo "--- DIRECTORY TREE of $dir ---" >> "$OUTPUT_FILE"
    tree "$dir" >> "$OUTPUT_FILE"
    echo -e "\n--- END OF DIRECTORY TREE ---\n" >> "$OUTPUT_FILE"
}

snapshot_sources() {
    local dir="$1"

    find "$dir" -type f | sort | while read -r file; do
        is_excluded "$file" && continue
        append_file "$file"
    done
}

snapshot_root_files() {
    for file in "${ROOT_FILES[@]}"; do
        [[ -f "$file" ]] || continue
        append_file "$file"
    done
}

#######################################
# Entry point
#######################################

main() {
    rm -f "$OUTPUT_FILE"

    for dir in "${SOURCE_DIRS[@]}"; do
        require_exists "$dir"
    done

    log "Creating codebase snapshot → $OUTPUT_FILE"

    for dir in "${SOURCE_DIRS[@]}"; do
        snapshot_tree "$dir"
        snapshot_sources "$dir"
    done

    snapshot_root_files

    log "Snapshot complete"
}

main "$@"
