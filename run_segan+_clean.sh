#!/bin/bash

CKPT_PATH="ckpt_segan+pase"

# please specify the path to your G model checkpoint
# as in weights_G-EOE_<iter>.ckpt
G_PRETRAINED_CKPT='weights_EOE_G-Generator-97301.ckpt'

# please specify the path to your folder containing
# noisy test files, each wav in there will be processed
TEST_FILES_PATH='data/noisy_testset_wav_16k'

# please specify the output folder where cleaned files
# will be saved
SAVE_PATH="synth_segan+pase"

python -u clean.py --g_pretrained_ckpt $CKPT_PATH/$G_PRETRAINED_CKPT \
	--test_files $TEST_FILES_PATH --cfg_file $CKPT_PATH/train.opts \
	--synthesis_path $SAVE_PATH --soundfile
