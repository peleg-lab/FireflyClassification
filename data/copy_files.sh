#!/bin/bash

# List of dataset folders
dataset_folders=(
    "/Volumes/peleg-group-1/Fireflies Citizen Science/_dataset_csvs/2020/Congaree/csvs"
    "/Volumes/peleg-group-1/Fireflies Citizen Science/_dataset_csvs/2020/GSMNP/csvs"
    "/Volumes/peleg-group-1/Fireflies Citizen Science/_dataset_csvs/2021/LincolnMA/csvs"
    "/Volumes/peleg-group-1/Fireflies Citizen Science/_dataset_csvs/2022/EstabrookMA/csvs"
    "/Volumes/peleg-group-1/Fireflies Citizen Science/_dataset_csvs/2021/EstabrookMA/csvs"
    "/Volumes/peleg-group-1/Fireflies Citizen Science/_dataset_csvs/2022/MuleshoeAZ/csvs"
    "/Volumes/peleg-group-1/Fireflies Citizen Science/_dataset_csvs/2022/Congaree/csvs"
    "/Volumes/peleg-group-1/Fireflies Citizen Science/_dataset_csvs/2022/BethanyBeachDE/csvs"
    "/Volumes/peleg-group-1/Fireflies Citizen Science/_dataset_csvs/2022/ForrestiGA/csvs"
    "/Volumes/peleg-group-1/Fireflies Citizen Science/_dataset_csvs/2021/PajaritoAZ/csvs"
    "/Volumes/peleg-group-2/Firefly_Experimental_Data/2024/LittleDryCreek_CO/GoPro/csvs"
    "/Volumes/peleg-group-2/Firefly_Experimental_Data/2024/RiverbendPonds_CO/GoPro/csvs"
    "/Volumes/peleg-group-2/Firefly_Experimental_Data/2023/Beanblossom_IN/GoPro/csvs"
)

# Remote destination folder (use your remote path, e.g., user@host:/remote/folder)
remote_destination="owma6084@fiji.colorado.edu:/Users/owma6084/FireFlyML/FireflyClassification/data/dataset_data"
ssh_key_path="~/.ssh/id_rsa"
# Loop over each folder in dataset_folders
for folder in "${dataset_folders[@]}"; do
    # Loop through each file in the folder
    for file in "$folder"/*; do
        # Only process files (not directories)
        if [[ -f "$file" ]]; then
            echo "Copying file: $file"

            # Use scp with the SSH key to copy the file to the remote directory
            scp -i "$ssh_key_path" "$file" "$remote_destination"

            if [[ $? -eq 0 ]]; then
                echo "Successfully copied $file"
            else
                echo "Failed to copy $file"
            fi
        fi
    done
done

echo "All files copied successfully!"
