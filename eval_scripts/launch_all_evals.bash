#!/bin/bash

# Create output directory for slurm logs if it doesn't exist
mkdir -p slurm_outputs

# Define evaluation sets and options
EVAL_SETS=("10" "spatial" "object" "goal")
#PATH_OPTIONS=("1")
PATH_OPTIONS=("0" "1")
MASK_OPTIONS=("0" "1")

# Loop through all combinations and launch jobs
for eval_set in "${EVAL_SETS[@]}"; do
    for use_path in "${PATH_OPTIONS[@]}"; do
        for use_mask in "${MASK_OPTIONS[@]}"; do
            # Skip invalid combinations if needed
            # For example, if path and mask shouldn't be used together, add a condition here
            if [[ $use_path -eq 0 && $use_mask -eq 0 ]]; then
                echo "Skipping job: no path and no mask for eval_set $eval_set"
                continue
            elif [[ $use_path -eq 0 && $use_mask -eq 1 ]]; then
                echo "Skipping job: no path and mask for eval_set $eval_set"
                continue
            fi
            
            # Create a descriptive job name
            path_str=""
            mask_str=""
            [[ $use_path -eq 1 ]] && path_str="_path"
            [[ $use_mask -eq 1 ]] && mask_str="_mask"
            
            job_name="libero_${eval_set}${path_str}${mask_str}"
            
            echo "Launching job: $job_name with params: $eval_set $use_path $use_mask"
            
            # Submit the job with parameters
            sbatch --job-name=$job_name eval_scripts/launch_eval.slurm $eval_set $use_path $use_mask
            
            # Optional: add a small delay between job submissions
            sleep 1
        done
    done
done

echo "All evaluation jobs submitted!"
