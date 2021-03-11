python3 main.py --batch_size 64 \
                --num_workers 8 \
                --cp "/content/drive/MyDrive/training/Quantized-classifier/" \
                --pretrained "./fp32_best.pth" \
                --mode "test_32to8"