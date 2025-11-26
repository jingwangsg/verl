# assert $HOME must exist
if test ! -d $HOME
    echo "Error: HOME directory does not exist!"
    exit 1
end


set -x tide_context_always_display true

# 自定义 tide context 显示用户名和IP
function _tide_item_context
    set -l ip (hostname -I | awk '{print $1}')
    if set -q SSH_TTY
        echo (set_color $tide_context_color_ssh)$USER@$ip(set_color normal)
    else if test $EUID -eq 0
        echo (set_color $tide_context_color_root)$USER@$ip(set_color normal)
    else
        echo (set_color $tide_context_color_default)$USER@$ip(set_color normal)
    end
end

if status is-interactive
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
. "$HOME/miniconda3/etc/fish/conf.d/conda.fish"
# if test -f /mnt/amlfs-03/shared/jingwang/miniconda3/bin/conda
#     eval /mnt/amlfs-03/shared/jingwang/miniconda3/bin/conda "shell.fish" "hook" $argv | source
# else
#     if test -f "/mnt/amlfs-03/shared/jingwang/miniconda3/etc/fish/conf.d/conda.fish"
#         . "/mnt/amlfs-03/shared/jingwang/miniconda3/etc/fish/conf.d/conda.fish"
#     else
#         set -x PATH "/mnt/amlfs-03/shared/jingwang/miniconda3/bin" $PATH
#     end
# end
# <<< conda initialize <<<
end

set -x PATH "$HOME/miniconda3/bin/:$PATH"
set -x TZ "Asia/Singapore"
set -gx GL_ROOT "$HOME/WORKSPACE/GearLaunchV2"
set -gx GL_PKG_DIR "$GL_ROOT/packages/gearlaunch"
set -gx GL_PYPROJECT "$GL_PKG_DIR/pyproject.toml"

function _delete_empty_args
    for arg in $argv
        if test -n "$arg"
            echo $arg
        end
    end
end

function gloga
    set num (default $argv[1] 50)
    git --no-pager log --oneline --decorate --graph --all --color | head -n $num
end


function knkill
    # if $argv == "-"
    #     knkill torchrun
    #     knkill debugpy
    #     knkill deepspeed
    #     knkill python
    #     knkill accelerate
    # end
    ps aux | grep "$argv" | grep -v grep | awk '{print $2}' | xargs kill -9
end

function rl
    readlink -f $argv
end

function nv
    nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv
end

function red_print
    echo -e "\033[31m$argv\033[0m"
end

function green_print
    echo -e "\033[32m$argv\033[0m"
end

function select_workflow
    set workflow_infos (ols $argv | grep Nodes)
    if test -z "$workflow_infos"
        red_print "No workflows found"
        return
    end
    
    # Convert multiline string to array and pipe to fzf
    printf '%s\n' $workflow_infos | fzf | awk '{print $2}'
end

function oexec
    # 使用 argparse 解析参数
    argparse -n oexec 't/task=' 'w/workflow=' 'home=' -- $argv; or return

    # 设置默认值和获取解析后的值
    set -l task (default $_flag_task "master")
    set -l workflow $_flag_workflow
    set -l home (default $_flag_home "amlfs-03/shared")

    # argparse 之后，$argv 自动包含未被解析的剩余参数
    set -l pass_args $argv
    set pool (osmo task list -a -c 1000 -w $workflow | grep RUNNING | head -n 1 | awk '{print $5}')
    if test "$pool" = "groot-gb200-01"
        set home "aws-lfs-01/shared"
    else if test "$pool" = "groot-h100-large-01" -o "$pool" = "groot-h100-large-02" -o "$pool" = "groot-h100-medium-01" -o "$pool" = "groot-h100-medium-02" -o "$pool" = "groot-h100-small-01" -o "$pool" = "groot-h100-small-02"
        set home "amlfs-03/shared"
    else if test "$pool" = "groot-l40-01" -o "$pool" = "groot-l40-02" -o "$pool" = "groot-l40-03"
        set home "amlfs-03/shared"
    else if test "$pool" = "groot-l40-04"
        set home "amlfs-04/home"
    else if test "$pool" = "groot-l40s-01"
        set home "lfs/home"
    end

    # Now pass only "clean" arguments to select_workflow
    if test -z $workflow
        set workflow (select_workflow $pass_args)
    end

    green_print "Exec into task $task in workflow $workflow"
    osmo workflow exec $workflow $task --entry "/mnt/$home/jingwang/WORKSPACE/GearLaunchV2/custom_bash"
end

function opool
    set pool_info (osmo pool list | grep ONLINE | fzf )
    set pool_name (echo $pool_info | awk '{print $1}')
    green_print "Set pool in profile to $pool_name"
    green_print $pool_info
    osmo profile set pool $pool_name
end

function otask
    osmo task list -c 2000 -s QUEUED -u jingwang@.com -v
end

function format
    if test (count $argv) -eq 0
        set cur_dir (pwd)
    else
        set cur_dir $argv[1]
    end
    green_print "Formating $cur_dir"
    if test -f "$GL_PYPROJECT"
        isort --settings-path $GL_PYPROJECT --dont-float-to-top --remove-redundant-aliases false -- $cur_dir
        black --config $GL_PYPROJECT $cur_dir
    else
        isort --dont-float-to-top --remove-redundant-aliases false -- $cur_dir
        black $cur_dir
    end

    chmod -R 777 $cur_dir &
end



function oport
    set port 5678
    set task "master"

    for i in (seq (count $argv))
        switch $argv[$i]
            case -p --port
                set port $argv[(math $i + 1)]
            case --job
                set job $argv[(math $i + 1)]
            case --task
                set task $argv[(math $i + 1)]
        end
    end

    if test -z $job
        set job (select_workflow)
    end

    green_print "Port forward task $task in workflow $job to localhost:$port"

    osmo workflow port-forward $job $task --port $port
end


function _parse_args_o
    set name debug
    set username jingwang
    set gpu 8
    set cpu 80
    set nodes 1
    set memory 1536Gi
    set storage 768Gi
    set image nvcr.io/nvidian/gear-training:latest
    set exec_timeout 4d
    set stdout 0
    set pool (osmo profile list | grep default | tail -n -1 | awk -F': ' '{print $2}')
    set cmd "echo 'Container ready for debugging'; sleep infinity"
    
    set num_argv (count $argv)
    for i in (seq $num_argv)
        switch $argv[$i]
            case --name
                set name $argv[(math $i + 1)]
                set argv[$i] ""
                set argv[(math $i + 1)] ""
            case --user
                set username $argv[(math $i + 1)]
                set argv[$i] ""
                set argv[(math $i + 1)] ""
            case -g --gpu
                set gpu $argv[(math $i + 1)]
                set argv[$i] ""
                set argv[(math $i + 1)] ""
            case -c --cpu
                set cpu $argv[(math $i + 1)]
                set argv[$i] ""
                set argv[(math $i + 1)] ""
            case -n --nodes
                set nodes $argv[(math $i + 1)]
                set argv[$i] ""
                set argv[(math $i + 1)] ""
            case -m --memory
                set memory $argv[(math $i + 1)]
                set argv[$i] ""
                set argv[(math $i + 1)] ""
            case -s --storage
                set storage $argv[(math $i + 1)]
                set argv[$i] ""
                set argv[(math $i + 1)] ""
            case -t --timeout
                set exec_timeout $argv[(math $i + 1)]
                set argv[$i] ""
                set argv[(math $i + 1)] ""
            case --stdout
                set stdout 1
                set argv[$i] ""
            case --image
                set image $argv[(math $i + 1)]
                set argv[$i] ""
                set argv[(math $i + 1)] ""
            case --pool
                set pool $argv[(math $i + 1)]
                set argv[$i] ""
                set argv[(math $i + 1)] ""
            case --cmd
                # read until next flag
                set cmd ""
                set j (math $i + 1)
                while test (math $num_argv + 1) -gt $j
                    if string match -q -- "-*" $argv[$j]
                        break
                    end
                    set cmd "$cmd$argv[$j] "
                    set argv[$j] ""
                    set j (math $j + 1)
                end
                set argv[$i] ""
        end
    end
    
    set kwargs (_delete_empty_args $argv)

    echo $name"<SEP>"$username"<SEP>"$gpu"<SEP>"$cpu"<SEP>"$nodes"<SEP>"$memory"<SEP>"$storage"<SEP>""$image""<SEP>"$stdout"<SEP>"$pool"<SEP>"$exec_timeout"<SEP>"$cmd"<SEP>""$kwargs"
end

function killjobs
    jobs -p | tail +1 | xargs kill -9
end

abbr kj "killjobs"
abbr rsz "resize"
abbr pypdb "python -m pdb -c continue"
abbr dbg "debugpy --listen 5678 --wait-for-client"

function ospec
    set full 0 

    for i in (seq (count $argv))
        switch $argv[$i]
            case --job
                set job $argv[(math $i + 1)]
                set argv[$i] ""
                set argv[(math $i + 1)] ""
            case --full
                set full 1
                set argv[$i] ""
        end
    end
    set argv (_delete_empty_args $argv)
    
    if test -z $job
        set job (select_workflow $argv)
    end

    set content (osmo workflow spec $job)

    if test $full -eq 1
        for line in (string split "\n" $content)
            echo $line
        end
    else
        echo "<--------------------- Resources ---------------------->"
        string split "\n" $content | shyaml get-value workflow.resources
        echo "<---------------------  Command  ---------------------->"
        set lines (string split "\n" $content | shyaml get-value workflow.groups.0.tasks.0.files.0.contents)
        for line in $lines
            if string match -q "#*" $line
                green_print $line
            else
                echo $line
            end
        end
        set num_task (string split "\n" $content | grep "path:" | wc -l)
        green_print "Number of tasks: $num_task"
    end
end

function ologs
    set worker "master"
    for i in (seq (count $argv))
        switch $argv[$i]
            case -j --job
                set job $argv[(math $i + 1)]
                set argv[$i] ""
                set argv[(math $i + 1)] ""
            case -w --worker
                set worker $argv[(math $i + 1)]
                set argv[$i] ""
                set argv[(math $i + 1)] ""
        end
    end
    set argv (_delete_empty_args $argv)

    if test -z $job
        set workflow_infos (osmo workflow list --status RUNNING PENDING  -c 1000 | grep jingwang | awk '{print $2,$3,$4,$5,$6}')
        for workflow_info in $workflow_infos
            set workflow_id (echo $workflow_info | awk '{print $1}')
            set workflow_time (echo $workflow_info | awk '{print $2,$3,$4,$5}')
            if test -f "$HOME/osmo_logs/$workflow_id/$worker.log"
                set last_line_no (cat $HOME/osmo_logs/$workflow_id/$worker.log | wc -l)
                green_print "$workflow_id [$workflow_time] -- $HOME/osmo_logs/$workflow_id/$worker.log:$last_line_no"
                tail -n 1 "$HOME/osmo_logs/$workflow_id/$worker.log"
            else
                red_print "$workflow_id -- logs not found"
            end
            echo "================================================="
        end
        return
    end

    set workflow_info (osmo workflow list --status RUNNING PENDING -c 1000 --name $job | grep jingwang | awk '{print $2,$3,$4,$5,$6}')
    set workflow_id (echo $workflow_info | awk '{print $1}')
    set workflow_time (echo $workflow_info | awk '{print $2,$3,$4,$5}')

    if test -f "$HOME/osmo_logs/$workflow_id/$worker.log"
        set last_line_no (cat $HOME/osmo_logs/$workflow_id/$worker.log | wc -l)
        green_print "$workflow_id [$workflow_time] -- $HOME/osmo_logs/$workflow_id/$worker.log:$last_line_no"
        tail -n 1 "$HOME/osmo_logs/$workflow_id/$worker.log"
    else
        red_print "$workflow_id -- logs not found"
        osmo workflow logs $workflow_id $worker
    end

end

function wologs
    watch -c "fish -c \"ologs $argv\""
end

function wtf
    set num (default $argv[1] 10)
    set workflow_infos (osmo workflow list --status FAILED | grep -v longrun | grep -v obackground | tail -n -$num | awk '{print $2,$3,$4,$5,$6}')
    set worker "master"

    for workflow_info in $workflow_infos
        set workflow_id (echo $workflow_info | awk '{print $1}')
        set workflow_time (echo $workflow_info | awk '{print $2,$3,$4,$5}')
        if test -f "$HOME/osmo_logs/$workflow_id/$worker.log"
            set last_line_no (cat $HOME/osmo_logs/$workflow_id/$worker.log | wc -l)
            red_print "$workflow_id [$workflow_time] -- $HOME/osmo_logs/$workflow_id/$worker.log:$last_line_no"
            # tail -n 3 "$HOME/osmo_logs/$workflow_id/$worker.log"
        else
            red_print "$workflow_id -- logs not found"
        end
        # echo "================================================="
    end
end

function orm
    set filename $argv[1]
    # shift argv
    set argv (string join " " $argv[2..-1])
    set filename (rl $filename)
    red_print "Remove $filename"

    orun -g 0 --memory 32Gi --storage 32Gi --cpu 32 --name rm \
        --cmd "mv $filename $filename~; rm -rf $filename~; echo '$filename removed'" $argv
end

function ochmod
    set filename "."
    if test (count $argv) -eq 1
        set filename $argv[1]
    end
    set filename (rl $filename)
    red_print "Change permission of $filename to 777"

    if test -d $filename
        orun -g 0 --memory 32Gi --storage 32Gi --cpu 32 --name chmod \
            --cmd "cd $filename; fd -H --no-ignore --exec bash -c 'chmod 777 {}; echo {}'; echo 'Permission of $filename changed to 777'" $argv
    else
        orun -g 0 --memory 32Gi --storage 32Gi --cpu 32 --name chmod \
            --cmd "chmod 777 $filename; echo 'Permission of $filename changed to 777'" $argv
    end
end

function default
  for val in $argv
    if test "$val" != ""
      echo $val
      break
    end
  end
end
set PATH $HOME/.local/bin $PATH

abbr wj "exec su - jingwang"

function toh1
    osmo profile set pool groot-h100-01
end

function toh2
    osmo profile set pool groot-h100-02
end

function tol1
    osmo profile set pool groot-l40-01
end

function tol2
    osmo profile set pool groot-l40-02
end

function tol3
    osmo profile set pool groot-l40-03
end

function tol1s
    osmo profile set pool groot-l40s-01
end

function pssh
    bash $HOME/WORKSPACE/GearLaunchV2/scripts/pssh.sh $argv
end

function pssh_show
    bash $HOME/WORKSPACE/GearLaunchV2/scripts/pssh_show.sh $argv
end

function checklog
    for log in $HOME/.osmo/$argv/*.log
        echo $log
        cat $log | grep "File "
    end
end

# function rm
#     set filename $argv[1]
#     set filename (rl $filename)
#     red_print "Remove $filename"
#     set abs_path (readlink -f $filename)

#     # read prompt
#     read -P "Are you sure you want to remove $abs_path? [y/N] " -n 1 -r

#     mv $filename $filename~
#     rm -rf $filename~
# end

function taillog
    # if has second args
    set line 10
    if test (count $argv) -eq 2
        set line $argv[2]
    end
    for log in $HOME/osmo_logs/$argv[1]/*.log
        echo $log
        tail -n $line $log
    end
end

function raystart
    bash $HOME/WORKSPACE/GearLaunchV2/scripts/raystart.sh $argv
end

function rayexec
    python $HOME/WORKSPACE/GearLaunchV2/scripts/rayexec.py $argv
end

function setup_cluster
    bash $HOME/WORKSPACE/GearLaunchV2/scripts/setup_cluster.sh $argv
end

function bad_node_detection
    # 如何提取出port 8300？
    # 提取 ray dashboard 进程参数中的 --port=8300
    set dashboard_port (ps aux | grep 'ray/dashboard/dashboard.py' | grep -v grep | grep -o -- '--port=[0-9]\+' | head -n 1 | sed 's/--port=//')
    echo "Ray dashboard port: $dashboard_port"
    RAY_ADDRESS="http://127.0.0.1:$dashboard_port" ray job submit --submission-id bad_node_detection_$(date +%s%N) \
        --working-dir $HOME/WORKSPACE/GearLaunchV2/scripts/ \
        -- python bad_node_detection.py $argv
end

function dryrun_on_nodes
    /usr/bin/python $HOME/WORKSPACE/GearLaunchV2/scripts/dryrun_on_nodes.py $argv
end

function readf
    python $HOME/WORKSPACE/GearLaunchV2/scripts/readf.py $argv
end


function gears5
    s5cmd --credentials-file ~/.gear/aws_credentials --endpoint-url https://pdx.s8k.io $argv
end

function cpi
    # copy in place
    set src $argv[1]
    set src_dir (dirname $src)
    set dst $argv[2]
    cp -rfv $src {$src_dir}/$dst
    chmod -R 777 {$src_dir}/$dst
end

function read_video_meta
    # read video meta data including duration, bit_rate, frame count, width, height
    set video_path $argv[1]
    # pretty print json with jq
    ffprobe -v error -show_entries format=duration,bit_rate -show_entries stream=width,height,nb_frames -of json $video_path | jq
end

function ray_stop
    for job_name in $argv
        ray job stop $job_name
        if test $DELETE -eq 1
            ray job delete $job_name
        end
    end
end

function dump_frame
    # dump video frame given timestamp
    # output path is optional, default output path is $video_path.frame.jpg
    set video_path $argv[1]
    set timestamp $argv[2]
    set output_path $argv[3]

    set video_name (basename $video_path)
    if test -z $output_path
        set output_path $video_name.frame.jpg
    end
    ffmpeg -i $video_path -ss $timestamp -vframes 1 $output_path
    echo $output_path
end


abbr ch "chmod -R 777"
abbr ot "osmo task list -a -c 1000 -w"
abbr ol "osmo workflow list -c 9999"

abbr c7 "chmod -R 777"


# clean all path related to torch in LD_LIBRARY_PATH
set TMP_PATH ""
for path in (string split ":" $LD_LIBRARY_PATH)
    if not string match -q "*torch*" $path
        if test -n "$TMP_PATH"
            set TMP_PATH "$TMP_PATH:$path"
        else
            set TMP_PATH "$path"
        end
    end
end
set -gx LD_LIBRARY_PATH $TMP_PATH

# clean PYTORCH_BUILD_VERSION
set -e PYTORCH_BUILD_VERSION

set HYDRA_FULL_ERROR 1
set -gx EDITOR vim
set -gx OMP_NUM_THREADS 4

if test -f $HOME/.cache/huggingface/token
    set -gx HF_TOKEN (cat $HOME/.cache/huggingface/token)
end

if test -d /mnt/amlfs-02/shared/ckpts/
    set -gx HF_HUB_CACHE /mnt/amlfs-02/shared/ckpts/
else if test -d /mnt/aws-lfs-01/shared/ckpts/
    set -gx HF_HUB_CACHE /mnt/aws-lfs-01/shared/ckpts/
end
set -gx GEAR_CREDENTIALS $HOME/.gear/data_credentials
set -gx WANDB_API_KEY (cat $HOME/.netrc | grep password | awk '{print $2}')

set -gx ORIG_HTTP_PROXY $HTTP_PROXY
set -gx ORIG_HTTPS_PROXY $HTTPS_PROXY
function with_proxy
  HTTP_PROXY=$ORIG_HTTP_PROXY HTTPS_PROXY=$ORIG_HTTPS_PROXY $argv
end

function tms
    resize
    # if claude/$argv is an existing tmux session, attach to it, otherwise create a new one
    set session_name (basename (pwd))
    # generate postfix using absolute path hash
    set postfix_hash (echo -n (pwd) | sha256sum | awk '{print $1}')
    set session_name $session_name\_$postfix_hash
    if tmux has-session -t $session_name
        tmux attach-session -t $session_name
    else
        tmux new-session -s $session_name
    end
end

set -gx CLAUDE_CODE_MAX_OUTPUT_TOKENS 200000

function oew
    green_print "WORKFLOW_ID: $WORKFLOW_ID"
    oexec -w $WORKFLOW_ID
end

function uvpipe
    uv pip install -e $argv[1] --config-settings editable_mode=compat --no-deps
end

function gl_install
    if test -d "$GL_PKG_DIR"
        uv pip install -e $GL_PKG_DIR --config-settings editable_mode=compat --no-deps
    else
        red_print "GearLaunch package directory not found: $GL_PKG_DIR"
        return 1
    end
end

function claude_glm
    ANTHROPIC_MODEL="glm-4.5" ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic ANTHROPIC_AUTH_TOKEN=70105358211d4d08b909a2dda962fe6b.W1mv8Z7fzBqeftu6 claude
end

function claude_vllm
    set model "glm45"
    if tmux has-session -t vllm_proxy > /dev/null 2>&1
        ANTHROPIC_MODEL=$model ANTHROPIC_BASE_URL=http://127.0.0.1:5000 ANTHROPIC_AUTH_TOKEN=sk-ant-1234 claude $argv
        return
    end
    set -l OSMO_EXEC "/osmo/usr/bin/osmo"
    if whoami | grep -q jingwang
        set workflow_id (su - jingwang -c "$OSMO_EXEC workflow list -c 9999 -s RUNNING | grep deploy | awk '{print \$2}' | head -n 1")
        echo $workflow_id
        tmux new-session -d -s vllm_proxy "su - jingwang -c \"$OSMO_EXEC workflow port-forward $workflow_id master --port 5000:4000\""
    else
        set workflow_id ($OSMO_EXEC workflow list -c 9999 -s RUNNING | grep deploy | awk '{print $2}' | head -n 1)
        echo $workflow_id
        tmux new-session -d -s vllm_proxy "$OSMO_EXEC workflow port-forward $workflow_id master --port 5000:4000"
    end
    echo "vllm proxy running at http://localhost:5000"
    ANTHROPIC_MODEL=$model ANTHROPIC_BASE_URL=http://127.0.0.1:5000 ANTHROPIC_AUTH_TOKEN=sk-ant-1234 claude $argv
end

function deploy_vllm
    cd $HOME/WORKSPACE/vllm_deploy && fish launch.fish
end

function ray_daemon
    set working_dir (pwd)
    set dashboard_port (ps aux | grep 'ray/dashboard/dashboard.py' | grep -v grep | grep -o -- '--port=[0-9]\+' | head -n 1 | sed 's/--port=//')
    echo "Ray dashboard port: $dashboard_port"
    cd $HOME/WORKSPACE/GearLaunchV2/scripts/
    RAY_ADDRESS="http://127.0.0.1:$dashboard_port" ray job submit --no-wait --working-dir . --submission-id "ray_daemon" --runtime-env-json '{"env_vars": {"RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING":"1", "RAY_DEDUP_LOGS": "0"}}' -- python ray_daemon.py
end

function ray_daemon_stop
    ray job stop "ray_daemon"
    ray job delete "ray_daemon"
end

function clone_gr00t
    GIT_LFS_SKIP_SMUDGE=1 with_proxy git clone https://gitlab-master.nvidia.com/GR00T/gr00t/gr00t.git
    PATH=/usr/bin:/usr/local/bin:$PATH uv venv --system-site-packages --no-managed-python
    . .venv/bin/activate.fish
    uv pip install tyro
    uvpipe gr00t
    uvpipe gr00t/external_dependencies/gear_cli/
    chmod -R 777 . &
    cd gr00t && git config core.fileMode false && cd ..
end

function clone_groot1
    GIT_LFS_SKIP_SMUDGE=1 with_proxy git clone https://gitlab-master.nvidia.com/GR00T/groot.git
    PATH=/usr/bin:/usr/local/bin:$PATH uv venv --system-site-packages --no-managed-python
    . .venv/bin/activate.fish
    uvpipe groot
    chmod -R 777 . &
    cd groot && git config core.fileMode false && cd ..
end

function ray_layoff
    bash $HOME/WORKSPACE/GearLaunchV2/scripts/ray_layoff.sh $argv
end

function ray_job_submit
    # if not found, query until dashboard_port is found
    while test -z $dashboard_port
        set dashboard_port (ps aux | grep 'ray/dashboard/dashboard.py' | grep -v grep | grep -o -- '--port=[0-9]\+' | head -n 1 | sed 's/--port=//')
        if test -z $dashboard_port
            sleep 1
        end
    end
    echo "Ray dashboard port: $dashboard_port"

    # 先手动找到 -- 的位置，分离选项和 entrypoint
    set -l separator_idx 0
    for i in (seq (count $argv))
        if test "$argv[$i]" = "--"
            set separator_idx $i
            break
        end
    end

    set -l options_args
    set -l entrypoint_args

    if test $separator_idx -eq 0
        # 没有找到 --，所有参数都是选项
        set options_args $argv
    else
        # 找到了 --，分离选项和 entrypoint
        for i in (seq (count $argv))
            if test $i -lt $separator_idx
                set options_args $options_args $argv[$i]
            else if test $i -gt $separator_idx
                set entrypoint_args $entrypoint_args $argv[$i]
            end
        end
    end

    # 解析选项部分（需要先将 options_args 复制到 argv，因为 argparse 会修改 argv）
    set argv $options_args
    argparse --ignore-unknown -n ray_job_submit 'submission-id=' 'skip-exists' -- $argv; or return
    set -l submission_id (default $_flag_submission_id "ray_submission")

    # argparse 处理后，$argv 只包含未识别的选项（即 ray job submit 的选项）
    set -l ray_options $argv

    # Check if job with same submission_id (without timestamp) already exists
    if set -q _flag_skip_exists
        # Get existing jobs with RUNNING or SUCCEEDED status
        set job_list_output (RAY_ADDRESS="http://127.0.0.1:$dashboard_port" ray job list 2>/dev/null)

        # Extract submission_ids from jobs with RUNNING or SUCCEEDED status
        for line in (echo $job_list_output | string split "JobDetails")
            # Check if this job has RUNNING or SUCCEEDED status
            if string match -q "*status=<JobStatus.RUNNING*" $line; or string match -q "*status=<JobStatus.SUCCEEDED*" $line
                # Extract submission_id from this job detail
                if string match -rq "submission_id='([^']+)'" $line
                    set existing_submission_id (string match -rg "submission_id='([^']+)'" $line)

                    # Remove timestamp suffix (last underscore followed by digits)
                    set existing_name (string replace -r '_[0-9]+$' '' $existing_submission_id)

                    # Compare with current submission_id
                    if test "$existing_name" = "$submission_id"
                        green_print "Job with submission_id '$submission_id' already exists, skipping submission."
                        green_print "Found: $existing_submission_id with RUNNING or SUCCEEDED status"
                        return 0
                    end
                end
            end
        end
    end

    # Add timestamp to submission_id
    set submission_id_with_timestamp {$submission_id}_(date +%s%N)

    green_print "Submitting job with submission_id: $submission_id_with_timestamp"

    # 构建完整的命令：ray job submit --submission-id <id> <ray_options> -- <entrypoint>
    if test (count $entrypoint_args) -gt 0
        RAY_ADDRESS="http://127.0.0.1:$dashboard_port" ray job submit --submission-id $submission_id_with_timestamp $ray_options -- $entrypoint_args
    else
        RAY_ADDRESS="http://127.0.0.1:$dashboard_port" ray job submit --submission-id $submission_id_with_timestamp $ray_options
    end
end

function addvip
    groupadd -g "1021" "jingwang" || true
    adduser --home "$HOME" --uid "1021" --gid "1021" --disabled-password --gecos "" "jingwang"
end

function torch28_install
    pssh --cmd "bash $HOME/WORKSPACE/GearLaunchV2/scripts/torch28_install.sh"
end


# quick touch
function qt 
    set FILENAME $argv[1]
    if test -f $FILENAME
        chmod 777 $FILENAME || true
    end
    mkdir -p $HOME/.tmp
    touch $HOME/.tmp/$FILENAME

    echo $HOME/.tmp/$FILENAME
    # if cursor exists
    if command -v cursor > /dev/null 2>&1
        cursor $HOME/.tmp/$FILENAME
    else
        vim $HOME/.tmp/$FILENAME
    end
end

function qp
    set FILENAME $HOME/.tmp/$argv[1]
    set argv $argv[2..-1]
    if test -f $FILENAME
        chmod 777 $FILENAME || true
    end
    python $FILENAME $argv
end


# --- set ca certificate ---
set -gx NODE_EXTRA_CA_CERTS "/etc/ssl/certs/npm-bundle.crt"

# --- unset proxy ---
set -gx no_proxy "127.0.0.1,localhost"
set -gx NO_PROXY "127.0.0.1,localhost"
set -e HTTP_PROXY
set -e HTTPS_PROXY
set -e http_proxy
set -e https_proxy
set -e ALL_PROXY
set -e all_proxy
