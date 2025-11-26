echo "groot-h100-large-01"
osmo workflow list -a -c 1000 -p groot-h100-large-01 -s RUNNING | grep cluster | awk '{print $2}' | xargs -I{} sh -c 'echo {}; gear ray list {}; osmo task list -a -c 9999 -w {} | grep nvidia | wc -l'

echo "groot-h100-large-02"
osmo workflow list -a -c 1000 -p groot-h100-large-02 -s RUNNING | grep cluster | awk '{print $2}' | xargs -I{} sh -c 'echo {}; gear ray list {}; osmo task list -a -c 9999 -w {} | grep nvidia | wc -l'

echo "groot-h100-medium-01"
osmo workflow list -a -c 1000 -p groot-h100-medium-01 -s RUNNING | grep cluster | awk '{print $2}' | xargs -I{} sh -c 'echo {}; gear ray list {}; osmo task list -a -c 9999 -w {} | grep nvidia | wc -l'

echo "groot-h100-medium-02"
osmo workflow list -a -c 1000 -p groot-h100-medium-02 -s RUNNING | grep cluster | awk '{print $2}' | xargs -I{} sh -c 'echo {}; gear ray list {}; osmo task list -a -c 9999 -w {} | grep nvidia | wc -l'

echo "groot-h100-small-01"
osmo workflow list -a -c 1000 -p groot-h100-small-01 -s RUNNING | grep cluster | awk '{print $2}' | xargs -I{} sh -c 'echo {}; gear ray list {}; osmo task list -a -c 9999 -w {} | grep nvidia | wc -l'

echo "groot-h100-small-02"
osmo workflow list -a -c 1000 -p groot-h100-small-02 -s RUNNING | grep cluster | awk '{print $2}' | xargs -I{} sh -c 'echo {}; gear ray list {}; osmo task list -a -c 9999 -w {} | grep nvidia | wc -l'