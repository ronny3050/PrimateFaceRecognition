
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/ronny3050/PrimateFaceRecognition">
    <img src="https://lh3.googleusercontent.com/VtgpdaLcoLwBLzGnxnUehm5u0faQpqoHrIwIp9p9DZTIU69dbZpi5oadz8lZlSKvzw=s180" alt="Logo" width="150" height="150">
  </a>

  <h3 align="center">PrimID: Face Recognition for Primates in the Wild</h3>

  <p align="center">
    <br />
    <a href="https://arxiv.org/abs/1804.08790"><strong>Explore the paper »</strong></a>
    <br />
    <br />
    <a href="https://play.google.com/store/apps/details?id=com.deb.debayan.primatefacerecognition&hl=en_US">View App</a>
    ·
    <a href="https://github.com/ronny3050/PrimateFaceRecognition/issues">Report Bug</a>
    ·
    <a href="https://github.com/ronny3050/PrimateFaceRecognition/issues">Request Feature</a>
  </p>
</p>


### Prerequisites

This project uses [Tensorflow](https://www.tensorflow.org/).

### Installation

Simply clone the repo
```sh
git clone https://github.com/ronny3050/PrimateFaceRecognition
```

<!-- USAGE EXAMPLES -->
## Usage

For training on primate face images:
```sh
python train.py --config_file config.py
```
<!-- CONFIG EXAMPLE -->
## Configuration
The `config.py` file contains all the configurations required for training. To train on your own primate dataset, simply change the `dataset_path` parameter to the root directory name. The root directory should contain subdirectories of primate individuals.

The partitions are defined in `splits` directory. Sample files are provided with this repository.


<!-- CONTRIBUTING -->
## Contributing

Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Debayan Deb - debdebay@msu.edu
