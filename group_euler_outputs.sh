#!/bin/bash

# This script takes all files from the euler_outputs folder, creates 1 subfolder per day, and moves the files
# into the appropriate subfolder in a sorted order (using the timestamp in the filename)

folder="euler_outputs"

# Filename pattern: <word>-<word>-<year>-<month>-<day>_<hour>-<minute>-<second>
# (sometimes there are 3 words)
# e.g. abstract-variance-2023-09-10_23-54-51

# Construct a list of all dates appearing in the filenames
filenames=$(ls -1 ${folder})
# remove the timestamps from the filenames
filenames_prefixes=$(echo "${filenames}" | cut -d'_' -f1)
# extract the dates from the filenames (starting from the end)
# extract <year>-<month>-<day> from the end of the filename
dates=$(echo "${filenames_prefixes}" | rev | cut -d'-' -f1-3 | rev)

# Create a subfolder for each date
for date in ${dates}; do
    mkdir -p "${folder}/${date}"
done

# Exclude any subfolders from folder
files=$(find euler_outputs -maxdepth 1 -type f)
# remove the folder from the path of each file in files
files=$(echo "${files}" | sed "s/${folder}\///g")
# exclude .DS_Store files
files=$(echo "${files}" | grep -v ".DS_Store")
# sort the files by time
files=$(echo "${files}" | sort -t'-' -k4,4 -k5,5 -k6,6)

# List all files in the folder, and sort them by timestamp (s.t. the oldest files are the first)
# -> move them in this order -> automatically orders them in the folder
for file in ${files}; do
    # extract the date from the filename
    date=$(echo "${file}" | cut -d'_' -f1 | rev | cut -d'-' -f1-3 | rev)
    # move the file into the appropriate subfolder
    mv "${folder}/${file}" "${folder}/${date}/${file}"
done