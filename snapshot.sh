#!/bin/bash

# The directory you want to create a snapshot of.
TARGET_DIR="src"

# Root-level README file
README_FILE="README.md"

# The name of the output file.
OUTPUT_FILE="codebase_snapshot.txt"

# --- Script starts here ---

# Check that the target directory exists.
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: The '$TARGET_DIR' directory was not found in the current location."
    echo "Please run this script from your project's root directory."
    exit 1
fi

# Check that README.md exists (non-fatal)
if [ ! -f "$README_FILE" ]; then
    echo "Warning: '$README_FILE' not found. It will be skipped."
fi

# Remove the old snapshot file if it exists.
rm -f "$OUTPUT_FILE"

echo "Creating a snapshot of the '$TARGET_DIR' directory..."
echo "The output will be saved to: $OUTPUT_FILE"

# 1. Add the directory tree structure.
echo "--- DIRECTORY TREE of $TARGET_DIR ---" > "$OUTPUT_FILE"
tree "$TARGET_DIR" >> "$OUTPUT_FILE"
echo -e "\n--- END OF DIRECTORY TREE ---\n" >> "$OUTPUT_FILE"

# 2. Add the content of each file in src/.
echo "--- CONTENT OF FILES in $TARGET_DIR ---" >> "$OUTPUT_FILE"
find "$TARGET_DIR" -type f -print0 | while IFS= read -r -d $'\0' file; do
    echo "--- START: $file ---" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo -e "\n--- END: $file ---\n" >> "$OUTPUT_FILE"
done

# 3. Add README.md (if present)
if [ -f "$README_FILE" ]; then
    echo "--- START: $README_FILE ---" >> "$OUTPUT_FILE"
    cat "$README_FILE" >> "$OUTPUT_FILE"
    echo -e "\n--- END: $README_FILE ---\n" >> "$OUTPUT_FILE"
fi

echo "Snapshot complete!"
echo "The full codebase snapshot has been saved to the file: $OUTPUT_FILE"
