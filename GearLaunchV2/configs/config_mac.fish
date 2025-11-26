if status is-interactive
    # Commands to run in interactive sessions can go here
end

function red_print
    echo -e "\033[31m$argv\033[0m"
end

function green_print
    echo -e "\033[32m$argv\033[0m"
end

function oexec
    # Default values
    set task "master"
    set home "amlfs-03/shared"

    # Make a copy of all original args
    set pass_args $argv

    # Parse out --task from $argv, store it in 'task'
    for i in (seq (count $argv))
        switch $argv[$i]
            case -t --task
                # Grab the next token for task
                set task $argv[(math $i + 1)]

                # Now remove --task and worker_7 (or whatever the value is) 
                # from pass_args so select_workflow doesn't see them
                set pass_args (string match -v -- "--task" $pass_args)
                set pass_args (string match -v -- $task $pass_args)
            case -w --workflow
                set workflow $argv[(math $i + 1)]

                set pass_args (string match -v -- "--workflow" $pass_args)
                set pass_args (string match -v -- $workflow $pass_args)
            case --home
                set home $argv[(math $i + 1)]
                set pass_args (string match -v -- "--home" $pass_args)
                set pass_args (string match -v -- $home $pass_args)
        end
    end
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

function gears5
    s5cmd --credentials-file ~/.gear/aws_credentials --endpoint-url https://pdx.s8k.io $argv
end

function killjobs
    jobs -p | tail +1 | xargs kill -9
end

function kill_remote
    ssh osmo_9000 "ps aux | grep cursor | grep -v grep | awk '{print \$2}' | xargs kill -9" || true
    ssh osmo_9002 "ps aux | grep cursor | grep -v grep | awk '{print \$2}' | xargs kill -9" || true
end

abbr ch "chmod -R 777"
abbr ot "osmo task list -a -c 1000 -w"
abbr ol "osmo workflow list -c 9999"
abbr c7 "chmod -R 777"
abbr kj "killjobs"
