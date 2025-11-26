
FILE_SYSTEM=$(df -h | grep /mnt/ | awk '{print $6}' | grep -v /home/ | sort | head -n 1)
echo $FILE_SYSTEM

if [[ "$FILE_SYSTEM" == *"/aws-lfs-01/"* ]]; then
    export SHARED=/mnt/aws-lfs-01/shared/
elif [[ "$FILE_SYSTEM" == *"/amlfs-04/"* ]]; then
    export SHARED=/mnt/amlfs-04/shared/
elif [[ "$FILE_SYSTEM" == *"/amlfs-01/"* ]]; then
    export SHARED=/mnt/amlfs-03/shared/
elif [[ "$FILE_SYSTEM" == *"/lfs/"* ]]; then
    export SHARED=/mnt/lfs/shared/
else
    echo "SHARED directory not found"
    exit 1
fi

dashboard_port=$(ps aux | grep 'ray/dashboard/dashboard.py' | grep -v grep | grep -o -- '--port=[0-9]\+' | head -n 1 | sed 's/--port=//')

mkdir -p $SHARED/scripts/
cp -rv $HOME/WORKSPACE/GearLaunchV2/scripts/ray_layoff.py $SHARED/scripts/tmp.py

RAY_ADDRESS="http://127.0.0.1:${dashboard_port}" ray job submit --submission-id "bad_node_detection" \
    -- python $SHARED/scripts/tmp.py --nnodes $1
ray job delete bad_node_detection
rm -rf $SHARED/scripts/tmp.py