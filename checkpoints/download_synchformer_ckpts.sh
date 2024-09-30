mkdir -p ./checkpoints/avclip_models/23-12-22T16-13-38

if [ ! -f "./checkpoints/avclip_models/23-12-22T16-13-38/epoch_best.pt" ]; then
    wget -P ./checkpoints/avclip_models/23-12-22T16-13-38 https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/23-12-22T16-13-38/checkpoints/epoch_best.pt
fi

if echo "4a566f2" && md5sum ./checkpoints/avclip_models/23-12-22T16-13-38/epoch_best.pt | cut -c 1-8 | grep -q "4a566f2"; then
    echo "Checksums match"
else
    echo "Checksums do not match. Deleting the file. Run the script again to download the file."
    rm ./checkpoints/avclip_models/23-12-22T16-13-38/epoch_best.pt
fi

mkdir -p ./checkpoints/sync_models/24-01-04T16-39-21

if [ ! -f "./checkpoints/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt" ]; then
    wget -P ./checkpoints/syncmodels/24-01-04T16-39-21 https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt
fi

if echo "54037d2" && md5sum ./checkpoints/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt | cut -c 1-8 | grep -q "54037d2"; then
    echo "Checksums match"
else
    echo "Checksums do not match. Deleting the file. Run the script again to download the file."
    rm ./checkpoints/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt
fi
