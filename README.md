# Datapipeline
Easy dataset preparation for speaking avatar model training.

## Installation
### Docker
Use docker to reduce pain in your butthole to the minimum:
```shell
# Build an image
$ docker build -t IMAGE_NAME .
# Run
$ docker run -it --gpus all --entrypoint=/bin/bash IMAGE_NAME 
```

To ensure that your image is built properly, run tests inside the container:
```shell
$ pytest tests
```

### Local (for development)
The tool is tested under Python3.9 / CUDA 11.3 machines only.

The most crucial packages are `torch`, `pytorch3d`, `transformers`,
so if you install them properly for your Python/CUDA
then you'll have 99% chance to use this tool.

#### Create Python venv
```shell
$ python3 -m venv dpl-venv
$ source dpl
$ ./scripts/install_deps.sh
```

#### Install jpegHDF5 plugin
Some computational nodes use `h5py` module to pack data into HDF5 containers.
In addition, we use [jpeg plugin](https://github.com/CARS-UChicago/jpegHDF5)
to save space when storing images.

You may choose one of the two options to build it under your machine.

**1. Stable, but with [Anaconda](https://docs.anaconda.com/)**
* Download and install Anaconda
* Run installation script
```shell
$ sudo bash scripts/build_jpeghdf5_anaconda.sh \
    thirdparty/jpegHDF5 \
    PATH/TO/ANACONDA
```

**2. Unstable, but without Anaconda**
* Ensure you have the latest versions of `libjpeg` and `libhdf5`
```shell
# On Ubuntu
$ sudo apt-get update -y
$ sudo apt-get install -y libjpeg-dev libhdf5-dev
```
* Run installation script
```shell
$ sudo bash scripts/build_jpeghdf5.sh thirdparty/jpegHDF5
```

#### Test your environment
Run to see if all dependencies installed correctly:
```shell
$ pytest tests
```


## Usage
### 1. Define computation graph
Like this:

```python
graph = [
    {
        'name': 'VideoToWavNode',
        'params': {
            'num_jobs': 32,
        }
    },
    {
        'name': 'Wav2vecNode',
        'params': {
            'device': "cuda:0",
        }
    }
]
```

Explanation: `name` is the class name of the computation node from `dpl/processor/nodes`,
and `params` is dict of its parameters. 

The graph above extracts audio in `.wav` format from each given video
and then computes `wav2vec` features.

More useful graphs can be found in `configs` folder.

### 2. Specify inputs
You already have your input data, right?
If so, just put roots to each data type in dict like this:

```python
inputs = {
    'video': 'root/path/to/videos',
    # If your computation graph takes more data types
    # then all of them should be specified here
    # 'wav': 'root/path/to/audio'
}
```

All avalable data types and their file extensions
can be found in `dpl/processor/datatypes.py`.

### 3. Run!
Save the graph and inputs to JSON-files 

```shell
$ python run.py \
    --graph "path/to/graph/file.json" \
    --inputs "path/to/inputs/file.json" \
    --output_dir "path/to/output/root"
```

## Backlog
* [x] Dockerize
* [ ] Nodes for visualizations
* [ ] Better logging
* [ ] Better error handling
* [ ] Parallel over GPUs
* [ ] Parametrizable image format: support JPEG and PNG
* [ ] Post-check output shapes for Numpy datatypes
* [ ] Support filtering input paths
* [ ] Datatypes with multiple allowed extensions: listdir outputs files matched to ALL extensions, but that's not desirable
* [ ] Mypy and format pre-commit check
* [ ] Tests
