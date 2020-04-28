# SICGAN

Code for [EECS 504](https://web.eecs.umich.edu/~ahowens/eecs504/w20/): Introduction to Computer Vision Project

## Requirements
- PyTorch 1.4.0
- [PyTorch 3D](https://github.com/facebookresearch/pytorch3d) 
- Cuda 10.1
- Cudnn 7.6

## Training

**Pixel2Mesh Baseline**
`python train_p2m.py --config-yml config/train_p2m.yml`

**Vanilla SICGAN**
`python train_p2m_gan.py --config-yml config/train_p2m_gan.yml`

**SICGAN with random noise**
`python train_p2m_randgan.py --config-yml config/train_p2m_randgan.yml`

## License
The SICGAN is realeased under [MIT License](https://github.com/dysdsyd/SICGAN/blob/master/LICENSE)




