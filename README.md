
# PHighly Undersampled MRI Reconstruction via a Single Posterior Sampling of Diffusion Models

- This repository is for our SSDM-MRI method, which enables a direct QSM reconstruction from MRI raw phases acquired at arbitrary orientations ([DOI link](https://doi.org/10.1016/j.media.2024.103160)).

- This code was built and tested on Win11 with RTX 4090, A4000, MacOS with M1 pro max, and a Centos 7.8 platform with Nvidia Tesla V100.

- **Major update, 19, March, 2025**: We have new and more user-friendly matlab wrappers for iQSM+/iQSM/iQFM/xQSM/xQSM+ reconstructions.

- **Minor Update**: For windows users: You will have to run `iQSM_fcns/ConfigurePython.m` first; modify variable `pyExec` (default:  
  `'C:\Users\CSU\anaconda3\envs\Pytorch\python.exe'`, % conda environment path (windows)), update the path with yours;

- **Minor Update**: see [Q&A about z_prjs](#qampa-about-z_prjs) for how to calculate vairbal zprjs;

---

## Content

- [Overview](#overview)
  - (1) [Overall Framework](#1-overall-framework)
  - (2) [Representative Results](#2-representative-results)
- [Manual](#manual)
  - [Requirements](#requirements)
  - [Quick Start](#quick-start)
  - [Q&A about z_prjs](#qampa-about-z_prjs)

---

## Overview

### (1) Overall Framework

Fig. 1: The overall structure of the proposed (a) Orientation-Adaptive Neural Network, which is constructed by incorporating (b) Plug-and-Play Orientation-Adaptive Latent Feature Editing (OA-LFE) blocks onto conventional deep neural networks. The proposed OA-LFE can learn the encoding of acquisition orientation vectors and seamlessly integrate them into the latent features of deep networks.

### (2) Representative Results

Fig. 2: Comparison of the original iQSM, iQSM-Mixed, and the proposed iQSM+ methods on (a) two simulated brains with different acquisition orientations, and (b) four in vivo brains scanned at multiple 3T MRI platforms.
