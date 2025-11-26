#!/bin/bash
# GPU Memory Cleaner Script
# Clean up GPU memory from various sources including:
# - PyTorch cached memory
# - Orphaned GPU processes  
# - Ray workers
# - Other CUDA applications

set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default options
SKIP_PROCESSES=false
AGGRESSIVE=false

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

print_step() {
    echo -e "${CYAN}🔧${NC} $1"
}

# Help function
show_help() {
    cat << EOF
GPU Memory Cleaner - Clean up GPU memory from various sources

Usage: $0 [OPTIONS]

Options:
    --skip-processes    Skip process killing
    --aggressive        Also kill processes found by keyword scan (can catch non-GPU owners; use with care)
    -h, --help         Show this help message

Examples:
    $0                  # Full GPU cleanup (force mode)
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-processes)
            SKIP_PROCESSES=true
            shift
            ;;
        --aggressive)
            AGGRESSIVE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Show GPU status
show_gpu_status() {
    echo -e "\n${CYAN}📊 GPU Memory Status:${NC}"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while read -r line; do
            # Split by comma and trim spaces
            gpu_id=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
            name=$(echo "$line" | cut -d',' -f2 | tr -d ' ')
            mem_used=$(echo "$line" | cut -d',' -f3 | tr -d ' ')
            mem_total=$(echo "$line" | cut -d',' -f4 | tr -d ' ')
            util=$(echo "$line" | cut -d',' -f5 | tr -d ' ')
            
            # Convert to GB
            mem_used_gb=$(echo "scale=1; $mem_used / 1024" | bc -l)
            mem_total_gb=$(echo "scale=1; $mem_total / 1024" | bc -l)
            echo "  GPU $gpu_id ($name): ${mem_used_gb}GB / ${mem_total_gb}GB ($util% util)"
        done
    else
        print_error "nvidia-smi not found"
    fi
}

kill_gpu_processes() {
    if [[ "$SKIP_PROCESSES" == true ]]; then
        return 0
    fi

    print_step "Step 1: Collecting GPU processes (nvidia-smi/fuser${AGGRESSIVE:+/ps})..."

    declare -A pid_sources
    local current_pid=$$
    local parent_pid=$PPID

    add_pid() {
        local pid="$1"
        local desc="$2"
        if [[ -z "$pid" || ! "$pid" =~ ^[0-9]+$ ]]; then
            return
        fi
        if [[ "$pid" == "1" || "$pid" == "$current_pid" || "$pid" == "$parent_pid" ]]; then
            return
        fi
        pid_sources["$pid"]+="${desc};"
    }

    # nvidia-smi compute apps
    if command -v nvidia-smi >/dev/null 2>&1; then
        while IFS=, read -r pid pname mem_used; do
            pid=$(echo "$pid" | tr -d ' ')
            pname=$(echo "$pname" | xargs)
            mem_used=$(echo "$mem_used" | tr -d ' ')
            [[ -z "$pid" || "$pid" == "No running processes found" ]] && continue
            add_pid "$pid" "nvidia-smi:${pname:-unknown}:${mem_used:-?}MiB"
        done < <(nvidia-smi --query-compute-apps=pid,process_name,memory.used --format=csv,noheader,nounits 2>/dev/null | sed '/No running processes found/d' || true)

        # nvidia-smi pmon (captures graphics/compute)
        while IFS=, read -r pid pname sm mem; do
            pid=$(echo "$pid" | tr -d ' ')
            pname=$(echo "$pname" | xargs)
            mem=$(echo "$mem" | tr -d ' ')
            [[ -z "$pid" || "$pid" == "-" ]] && continue
            add_pid "$pid" "pmon:${pname:-unknown}:${mem:-?}MiB"
        done < <(nvidia-smi pmon -c 1 2>/dev/null | awk 'NR>2 {print $2","$5","$6","$10}' || true)
    else
        print_warning "nvidia-smi not available; skipping direct GPU process lookup"
    fi

    # Processes holding /dev/nvidia* (fuser)
    if command -v fuser >/dev/null 2>&1; then
        shopt -s nullglob
        for device_path in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia[0-9]*; do
            [[ -e "$device_path" ]] || continue
            while read -r pid; do
                add_pid "$pid" "fuser:${device_path##*/}"
            done < <(fuser -v "$device_path" 2>/dev/null | awk 'NR>1 {print $2}' || true)
        done
        shopt -u nullglob
    else
        print_warning "fuser command not available; install psmisc for deeper cleanup"
    fi

    # lsof fallback if available
    if command -v lsof >/dev/null 2>&1; then
        while read -r pid; do
            add_pid "$pid" "lsof:/dev/nvidia*"
        done < <(lsof -w -t /dev/nvidia* 2>/dev/null || true)
    fi

    # Broad keyword sweep (opt-in)
    if [[ "$AGGRESSIVE" == true ]]; then
        while IFS=, read -r pid cmd; do
            add_pid "$pid" "ps:${cmd}"
        done < <(ps -eo pid,cmd | grep -iE "cuda|gpu|torch|deepspeed|accelerate|jax|tensorflow|nccl|nvidia|ray|triton|llama|inference" | grep -v grep | awk '{pid=$1; $1=""; sub(/^ /,""); print pid "," substr($0,1,120)}' || true)
    fi

    local pids=()
    for pid in "${!pid_sources[@]}"; do
        pids+=("$pid|${pid_sources[$pid]}")
    done

    echo "  ℹ️  Found ${#pids[@]} GPU-related processes"

    if [[ ${#pids[@]} -eq 0 ]]; then
        print_status "No GPU processes to kill"
        return 0
    fi

    local killed_count=0
    local total=${#pids[@]}
    # Skip patterns to protect control-plane processes
    local skip_patterns=("codex" "osmo" "sshd" "ssh" "tmux" "bash" "fish" "login" "agetty" "systemd" "journald" "dbus" "sudo" "socat" "fzf")

    for entry in "${pids[@]}"; do
        local pid=${entry%%|*}
        local desc=${entry#*|}

        local cmdline
        cmdline=$(ps -p "$pid" -o cmd= 2>/dev/null | head -n1 || true)

        local skip=false
        for pat in "${skip_patterns[@]}"; do
            if [[ "$cmdline" =~ $pat ]]; then
                skip=true
                break
            fi
        done

        if [[ "$skip" == true ]]; then
            echo "    ℹ️  Skipping PID $pid ($cmdline)"
            continue
        fi

        if kill -9 "$pid" 2>/dev/null; then
            echo "    ✅ Killed PID $pid (${desc:0:120})${cmdline:+ [$cmdline]}"
            ((killed_count++))
        else
            print_warning "Failed to kill PID $pid (${desc:0:120})${cmdline:+ [$cmdline]}"
        fi
    done

    print_status "Killed $killed_count/$total GPU-related processes"
}

# Main function
main() {
    echo "🚀 GPU Memory Cleaner Starting..."
    echo "=================================================="
    
    # Show initial GPU status
    show_gpu_status

    # Kill processes reported by multiple sources
    kill_gpu_processes
    
    echo ""
    echo "=================================================="
    print_status "GPU Memory Cleaner Complete!"
    
    # Show final GPU status
    show_gpu_status
}

# Check dependencies
check_dependencies() {
    local missing=()

    for cmd in bc; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing+=("$cmd")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        print_error "Missing required commands: ${missing[*]}"
        print_error "Please install the missing packages before running this script."
        exit 1
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        print_warning "nvidia-smi not found; GPU status and nvidia-smi based cleanup will be limited"
    fi

    if [[ "$SKIP_PROCESSES" == false ]] && ! command -v fuser >/dev/null 2>&1; then
        print_warning "fuser not available - GPU device checking will be limited"
    fi

    if [[ "$SKIP_PROCESSES" == false ]] && ! command -v lsof >/dev/null 2>&1; then
        print_warning "lsof not available - /dev/nvidia* handle detection will be limited"
    fi

    if [[ "$AGGRESSIVE" == true ]]; then
        print_info "Aggressive mode enabled: keyword-based scan will also be killed (can hit unrelated processes)."
    fi
}

# Run the script
check_dependencies
main
