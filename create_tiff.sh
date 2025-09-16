#!/bin/bash

# Usage: ./png2tif_all.sh /path/to/folder
# Converts all PNGs in the given folder to LZW-compressed TIFFs

if [ $# -lt 1 ]; then
    echo "Usage: $0 /path/to/folder"
    exit 1
fi

folder="$1"

if [ ! -d "$folder" ]; then
    echo "Error: $folder is not a directory"
    exit 1
fi

for file in "$folder"/*.png; do
    # Skip if no PNGs
    [ -e "$file" ] || continue
    
    base="${file%.*}"
    output="${base}.tif"

    echo "Converting $file → $output"
    magick "$file" -compress LZW "$output"
done

echo "All PNG files in $folder have been converted to LZW-compressed TIFFs."
