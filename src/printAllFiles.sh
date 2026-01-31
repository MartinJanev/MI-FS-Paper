#!/bin/bash

# Output file where all contents will be saved
output_file="all_contents.txt"

# Empty the output file if it exists
> "$output_file"

# Loop through all files in the current directory and subdirectories
# Loop through all files in the current directory and subdirectories
find . -type f \
    ! -path "*/.*" \
    ! -path "*/CMS_Open_Payements2024/*" \
    ! -path "*/__pycache__/*" \
    ! -path "*/data/*" \
    ! -path "*/results/*" \
    ! -path "*/output/*" \
    ! -path "*/venv/*" \
    ! -path "*/env/*" \
    ! -path "*/.venv/*" \
    ! -path "*/node_modules/*" \
    ! -path "*/.git/*" \
    ! -name "*.pyc" \
    ! -name "*.pyo" \
    ! -name "*.parquet" \
    ! -name "*.pyd" \
    ! -name "*.so" \
    ! -name "*.dll" \
    ! -name "*.exe" \
    ! -name "*.db" \
    ! -name "*.sqlite" \
    ! -name "*.log" \
    ! -name "*.jpg" \
    ! -name "*.jpeg" \
    ! -name "*.png" \
    ! -name "*.gif" \
    ! -name "*.ico" \
    ! -name "*.pdf" \
    ! -name "*.zip" \
    ! -name "*.tar.gz" \
    ! -name "*.md" \
    -o -name "README.md" \
    | while read -r file; do
    echo "===== File: $file =====" >> "$output_file"
    cat "$file" >> "$output_file"
    echo -e "\n" >> "$output_file"
done

# Include specific files from the parent directory
for file in ../*.md ../*.toml ../.gitignore; do
    if [ -f "$file" ]; then
        echo "===== File: $file =====" >> "$output_file"
        cat "$file" >> "$output_file"
        echo -e "\n" >> "$output_file"
    fi
done
echo "All contents have been written to $output_file"
