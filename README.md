
# Speech Enhancement Generative Adversarial Network in PyTorch
## Getting started

### Requirements
Python 3.6.9 or greater is required with the following packages
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
audioread==2.1.9
yapf==0.31.0
setuptools==54.2.0
```
Ahoprocessing tools (`ahoproc_tools`) is also needed, and the public repo is found [here](https://github.com/santi-pdp/ahoproc_tools).
### Installation
#### Clone repository
Clone the GitHub repo using
```bash
git clone https://github.com/mansoorcheema/segan_pytorch.git
```
It will fetch the code at the current path and create a directory `segan_pytorch`.
#### Install requirements
The requirements can be installed individually using `pip install <package-name> <package-version>` (preferred) or install all together using requirements.txt in the cloned repository with the command:
```bash
cd segan_pytorch
pip install -r requirements.txt
```
### Dataset
The speech enhancement dataset used in this work [(Valentini et al. 2016)](http://ssw9.net/papers/ssw9_PS2-4_Valentini-Botinhao.pdf) can be found in [Edinburgh DataShare](http://datashare.is.ed.ac.uk/handle/10283/1942). After downloading the noisy and clean datasets convert the wav files from 48 khz to 16 khz using the following steps. For simplicity, extract the downloaded zipped files in the data folder(create new folder) in the main directory. 

Move to the directory where you downloaded the data.
```bash
cd data
```
Create a folder to store the downsampled clean training data
```bash
mkdir -p clean_trainset_wav_16k 
```
Move to the directory containing the clean wav files, assuming they were extracted to `clean_trainset_wav` and downsample to 16 khz using the following commands
```bash
cd clean_trainset_wav
ls *.wav | while read name; do
    sox $name -r 16k ../clean_trainset_wav_16k/$name
done
cd ..
```
Now the downsampled wav clean training files should be found in `clean_trainset_wav_16k` directory. Follow the same approach to downsample the remaining parts. 

## Models

We extented SEGAN+, an improved version of SEGAN [1], and introduce the following Models
* Sinc Convolution [3] based discriminator
* PASE [4] based Discriminator
* Baseline SEGAN+ [1]


Both Models use standard SEGAN Generator

> **_NOTE:_**  A sinc convolution based Generator and Discrimininator GAN is accessible at a separate branch https://github.com/mansoorcheema/segan_pytorch/tree/sinc_generator.

### SEGAN+ Generator

![SEGAN+_G](assets/segan+.png)

### Sinc Convolution Discriminator
![sinc_disc](https://user-images.githubusercontent.com/10983181/149785311-5d1ee19b-d87f-4574-8490-43e7b390717c.png)
### PASE Discriminator
![pase_disc](https://user-images.githubusercontent.com/10983181/149784688-d1c25049-28f4-4359-baab-6c8c86a2784e.png)

## Audio Samples
A few enhanced samples providing a qualitative comparison of the [speech enhancement models](#Models) are provided below:

|Noisy| SEGAN+      | SEGAN+ Sinc | SEGAN+ Sinc*| SEGAN+ PASE  |
|---| ----------- | ----------- |----------- |----------- |
|[Play](https://raw.githubusercontent.com/mansoorcheema/segan_pytorch/master/enhanced_samples/p232_006-noisy.wav)| [Play](https://raw.githubusercontent.com/mansoorcheema/segan_pytorch/master/enhanced_samples/p232_006-segan+.wav)      |[Play](https://raw.githubusercontent.com/mansoorcheema/segan_pytorch/master/enhanced_samples/p232_006-sinc-disc.wav)        |[Play](https://raw.githubusercontent.com/mansoorcheema/segan_pytorch/master/enhanced_samples/p232_006-sinc_disc_and_gen.wav)   |[Play](https://raw.githubusercontent.com/mansoorcheema/segan_pytorch/master/enhanced_samples/p232_006-pase-disc.wav) |
|[Play](https://raw.githubusercontent.com/mansoorcheema/segan_pytorch/master/enhanced_samples/p257_430-noisy.wav)| [Play](https://raw.githubusercontent.com/mansoorcheema/segan_pytorch/master/enhanced_samples/p257_430-segan+.wav)     | [Play](https://raw.githubusercontent.com/mansoorcheema/segan_pytorch/master/enhanced_samples/p257_430-sinc-disc.wav)         |[Play](https://raw.githubusercontent.com/mansoorcheema/segan_pytorch/master/enhanced_samples/p257_430-sinc_disc__and_gen.wav)  |[Play](https://raw.githubusercontent.com/mansoorcheema/segan_pytorch/master/enhanced_samples/p257_430-pase-disc.wav)  |
> **_NOTE:_**  * Generator + Discriminator.
> 
## Training
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

## Enhancement
```
python clean.py --g_pretrained_ckpt ckpt_segan+/<weights_ckpt_for_G> \
		--cfg_file ckpt_segan+/train.opts --synthesis_path enhanced_results \
		--test_files data/noisy_testset --soundfile
```

Read `run_segan+_clean.sh` for more guidance.


## Credits:

1. [SEGAN: Speech Enhancement Generative Adversarial Network (Pascual et al. 2017)](https://arxiv.org/abs/1703.09452)
2. [Language and Noise Transfer in Speech Enhancement GAN (Pascual et al. 2018)](https://arxiv.org/abs/1712.06340)
3. [Speaker Recognition from Raw Waveform with SincNet(Ravanelli, et al. 2018)](https://arxiv.org/abs/1808.00158)
4. [Learning Problem-agnostic Speech Representations from Multiple Self-supervised Tasks(Pascual et al. 2019)](https://arxiv.org/abs/1904.03416)
 
