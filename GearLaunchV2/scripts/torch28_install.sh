#!/bin/bash
set -ex

export PATH=/usr/bin:/usr/local/bin:$PATH
unset PIP_CONSTRAINT

# Remove/backup old ffmpeg binaries to avoid PATH conflicts
if [ -f /usr/local/bin/ffmpeg ]; then
    echo "Found old ffmpeg at /usr/local/bin/ffmpeg, backing up..."
    mv /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg.old
fi

if [ -f /usr/local/bin/ffprobe ]; then
    mv /usr/local/bin/ffprobe /usr/local/bin/ffprobe.old
fi

if [ -f /usr/local/bin/ffplay ]; then
    mv /usr/local/bin/ffplay /usr/local/bin/ffplay.old
fi

# Try to remove any old ffmpeg package
apt-get remove -y ffmpeg || true

apt-get update
apt-get install -y libva2 libva-drm2 libva-x11-2 libdrm2 libvpx9 libaom3 libfdk-aac2 libx264-164 libx265-199

dpkg -i /mnt/aws-lfs-01/shared/collections/deb/ffmpeg-gb200_7.1-gb200-1_arm64.deb

which ffmpeg
ffmpeg -version | head -1

pip install /mnt/aws-lfs-01/shared/collections/wheel/torch-2.8.0/torch*.whl /mnt/aws-lfs-01/shared/collections/wheel/torch-2.8.0/flash*.whl torchvision==0.23.0