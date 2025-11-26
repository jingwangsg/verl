set -ex

apt-get update && apt-get install -y \
    openssh-server \
    sudo &&
    rm -rf /var/lib/apt/lists/*

mkdir -p /var/run/sshd
# Backup sshd_config before modifying
SSHD_CONFIG="/etc/ssh/sshd_config"

# Configure SSH to allow passwordless root login
echo "Configuring SSH for passwordless root login..."
sed -i '/^#\?PermitRootLogin/s/.*/PermitRootLogin yes/' "$SSHD_CONFIG"
sed -i '/^#\?PasswordAuthentication/s/.*/PasswordAuthentication yes/' "$SSHD_CONFIG"
sed -i '/^#\?PermitEmptyPasswords/s/.*/PermitEmptyPasswords yes/' "$SSHD_CONFIG"
sed -i '/^#\?PubkeyAuthentication/s/.*/PubkeyAuthentication yes/' "$SSHD_CONFIG"
sed -i '/^#\?PubkeyAcceptedKeyTypes/s/.*/PubkeyAcceptedKeyTypes ssh-ed25519,ssh-ed25519-cert-v01@openssh.com/' "$SSHD_CONFIG"

# Set empty password for root
passwd -d root
/usr/sbin/sshd -D

