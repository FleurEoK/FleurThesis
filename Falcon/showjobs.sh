#!/bin/bash
# Get all the jobs that are pending
squeue -u 20204130 > jobs_temp.txt

# Show the jobs
echo "== Jobs =="
echo "===================================================================================="
cat jobs_temp.txt

# Get the job ids and Status (JOBID, PARTITION, NAME, USER, ST, TIME, NODES, NODELIST(REASON))
awk '{print $1}' jobs_temp.txt > job_ids.txt

echo "===================================================================================="
# echo ""
# echo "== Job IDs =="
# echo "==============================================="
# # Get the status of the jobs
# for job_id in $(cat job_ids.txt)
# do
    
#     # Skip if the line is the header
#     if [ $job_id == "JOBID" ]; then
#         continue
#     fi
#     echo "Job ID: $job_id"
#     echo "---------------"

#     # Get the status of the job
#     job_status=$(squeue -j $job_id | awk 'NR>1 {print $5}')

#     # If status is PD, echo "Pending..."
#     if [ $job_status == "PD" ]; then
#         echo "Status: Pending..."
#     else
#         sstat --format=AveCPU,MinCPU,AveRSS -j $job_id
#     fi
#     echo "==============================================="
# done

# Remove the temporary files
rm jobs_temp.txt
rm job_ids.txt