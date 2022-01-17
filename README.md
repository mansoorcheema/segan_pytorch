# Speech Enhancement Generative Adversarial Network in PyTorch

### Requirements

```
SoundFile==0.10.2
scipy==1.1.0
librosa==0.6.1
h5py==2.8.0
numba==0.38.0
torch==0.4.1
matplotlib==2.2.2
numpy==1.14.3
pyfftw==0.10.4
tensorboardX==1.4
torchvision==0.2.1
```
Ahoprocessing tools (`ahoproc_tools`) is also needed, and the public repo is found [here](git@github.com:santi-pdp/ahoproc_tools.git).

### Audio Samples

Latest denoising audio samples with baselines can be found in the [segan+ samples website](http://veu.talp.cat/seganp/). SEGAN is the vanilla SEGAN version (like the one in TensorFlow repo), whereas SEGAN+ is the shallower improved version included as default parameters of this repo.

The voicing/dewhispering audio samples can be found in the [whispersegan samples website](http://veu.talp.cat/whispersegan). Artifacts can now be palliated a bit more with `--interf_pair` fake signals, more data than the one we had available (just 20 mins with 1 speaker per model) and longer training session by iterating more than `100 epoch`.

## Models

We extented SEGAN+, an improved version of SEGAN [1], and introduce following Models
* Sinc Convolution based discriminator
* PASE based Discriminator and standard SEGAN+ discriminator
Both Models share standard SEGAN Generator

### SEGAN+ Generator

![SEGAN+_G](assets/segan+.png)

### Sinc Convolution Discriminator
![sinc_disc](https://user-images.githubusercontent.com/10983181/149785311-5d1ee19b-d87f-4574-8490-43e7b390717c.png)
### PASE Discriminator
![pase_disc](https://user-images.githubusercontent.com/10983181/149784688-d1c25049-28f4-4359-baab-6c8c86a2784e.png)

### Usage
To train these models, the following command should be ran. For using **sinc convolution** for discriminator, provide argument `--sinc_conv`:
```

python -u train.py --save_path ckpt_segan+pase \
                   --clean_trainset data/clean_trainset_wav_16k \
		   --noisy_trainset data/noisy_trainset_wav_16k \
		   --cache_dir data_tmp --no_train_gen \
		   --batch_size 50 --no_bias --sinc_conv
```

Similarly provide `--pase_disc` for using **PASE** based discriminator:
```

python -u train.py --save_path ckpt_segan+pase \
                   --clean_trainset data/clean_trainset_wav_16k \
		   --noisy_trainset data/noisy_trainset_wav_16k \
		   --cache_dir data_tmp --no_train_gen \
		   --batch_size 50 --no_bias --pase_disc
```
Read `run_segan+_train.sh` for more guidance. This will use the default parameters to structure both G and D, but they can be tunned with many options. For example, one can play with `--d_pretrained_ckpt` and/or `--g_pretrained_ckpt` to specify a departure pre-train checkpoint to fine-tune some characteristics of our enhancement system, like language, as in [2].

Cleaning files is done by specifying the generator weights checkpoint, its config file from training and appropriate paths for input and output files (Use `soundfile` wav writer backend (recommended) specifying the `--soundfile` flag):

```
python clean.py --g_pretrained_ckpt ckpt_segan+/<weights_ckpt_for_G> \
		--cfg_file ckpt_segan+/train.opts --synthesis_path enhanced_results \
		--test_files data/noisy_testset --soundfile
```

Read `run_segan+_clean.sh` for more guidance.


### Credits:

1. [SEGAN: Speech Enhancement Generative Adversarial Network (Pascual et al. 2017)](https://arxiv.org/abs/1703.09452)
2. [Language and Noise Transfer in Speech Enhancement GAN (Pascual et al. 2018)](https://arxiv.org/abs/1712.06340)
3. [Learning Problem-agnostic Speech Representations from Multiple Self-supervised Tasks(Pascual et al. 2019)](https://arxiv.org/abs/1904.03416)
4. [Speaker Recognition from Raw Waveform with SincNet(Ravanelli, et al. 2018)](https://arxiv.org/abs/1808.00158)
 
