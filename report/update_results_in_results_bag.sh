# The script loops through all the folders in the results_bag folder (except the blacklisted folders) and updates the
# results in the results_bag folder using files from the results folder (if they are newer). Updating the files means
# taking the correct file from the correct model (model is indicated by an intermediate folder) and copying it to the
# results_bag folder.

blacklist_folders=( "BACKUP-09-13" "OLD-BAG" )
results_folder="../results"
results_bag_folder="../results_bag"

for folder in $results_bag_folder/*; do
    if [[ -d $folder ]]; then
        folder_name=$(basename $folder)
        if [[ ! " ${blacklist_folders[@]} " =~ " ${folder_name} " ]]; then
            for model in $folder/*; do
                if [[ -d $model ]]; then
                    model_name=$(basename $model)
                    for file in $model/*; do
                        if [[ -f $file ]]; then
                            file_name=$(basename $file)
                            if [[ -f $results_folder/$model_name/$file_name ]]; then
                                if [[ $file -ot $results_folder/$model_name/$file_name ]]; then
                                    echo "Updating $file with $results_folder/$model_name/$file_name"
                                    cp $results_folder/$model_name/$file_name $file
                                fi
                            fi
                        fi
                    done
                fi
            done
        fi
    fi
done