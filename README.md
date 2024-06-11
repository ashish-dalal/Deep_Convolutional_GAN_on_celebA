# Deep Convolutional GAN on CelebA

Deep Convolutional Generative Adversarial Network trained on facial images from the CelebA dataset.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains the implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) that is trained on the CelebA dataset to generate realistic images of celebrity faces.

## Features

- Implementation of DCGAN architecture.
- Training script for the CelebA dataset.
- Pre-trained models.
- Visualization of generated images.
- TensorBoard support for monitoring training progress.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ashish-dalal/Deep_Convolutional_GAN_on_celebA.git
    cd Deep_Convolutional_GAN_on_celebA
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the CelebA dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and extract it into the `data/` directory.

## Usage

1. To train the DCGAN, run:
    ```bash
    python train.py
    ```

2. To visualize the results, you can use TensorBoard:
    ```bash
    tensorboard --logdir=tensorboard-summary/
    ```

## Results

Generated images will be saved in the `results/` directory. You can find sample generated images and training logs in this directory.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
