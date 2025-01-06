#!/bin/bash

# python -m flash_vstream.serve.cli_video_stream \
#     --model-path /home/vault/b232dd/b232dd16/Flash-VStream/Flash-VStream-7b \
#     --video-file /home/vault/b232dd/b232dd16/Flash-VStream/assets/needle_32.mp4 \
#     --conv-mode vicuna_v1 --temperature 0.0 \
#     --video_max_frames 1200 \
#     --video_fps 1.0 --play_speed 1.0 \
#     --log-file realtime_cli.log

python -m flash_vstream.serve.cli_video_stream \
    --model-path /home/vault/b232dd/b232dd21/vlm/Flash-VStream/Flash-VStream-7b \
    --video-file /home/vault/b232dd/b232dd21/vlm/Flash-VStream/assets/needle_32.mp4 \
    --conv-mode vicuna_v1 --temperature 0.0 \
    --video_max_frames 1200 \
    --video_fps 1.0 --play_speed 1.0 \
    --log-file realtime_cli.log
