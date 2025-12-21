#!/bin/bash

# The directory you want to create a snapshot of.
TARGET_DIR="src"

# The name of the output file.
OUTPUT_FILE="codebase_snapshot.txt"

# --- Script starts here ---

# First, check if the target directory exists.
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: The '$TARGET_DIR' directory was not found in the current location."
    echo "Please run this script from your project's root directory."
    exit 1
fi

# Remove the old snapshot file if it exists to start fresh.
rm -f "$OUTPUT_FILE"

echo "Creating a snapshot of the '$TARGET_DIR' directory..."
echo "The output will be saved to: $OUTPUT_FILE"

# 1. Add the directory tree structure to the snapshot file.
# The '>' operator creates the file and adds the first content.
echo "--- DIRECTORY TREE of $TARGET_DIR ---" > "$OUTPUT_FILE"
tree "$TARGET_DIR" >> "$OUTPUT_FILE"
echo -e "\n--- END OF DIRECTORY TREE ---\n" >> "$OUTPUT_FILE"

# 2. Add the content of each file within the target directory.
# The '>>' operator appends the rest of the content to the file.
echo "--- CONTENT OF FILES in $TARGET_DIR ---" >> "$OUTPUT_FILE"
find "$TARGET_DIR" -type f -print0 | while IFS= read -r -d $'\0' file; do
    echo "--- START: $file ---" >> "$OUTPUT_FILE"
    # Append the content of the file.
    cat "$file" >> "$OUTPUT_FILE"
    # Add newlines for better separation between files.
    echo -e "\n--- END: $file ---\n" >> "$OUTPUT_FILE"
done

echo "Snapshot complete!"
echo "The full codebase snapshot has been saved to the file: $OUTPUT_FILE"