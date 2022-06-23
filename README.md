# Datapipeline
Easy dataset preparation for speaking avatar model training.

## Installation
Now it depends on Python3.9 and CUDA 11.3 but it's not really a necessity. 

The most crucial packages are `torch`, `pytorch3d`, `transformers`,
so if you install them properly with your Python/CUDA
then you'll 99% sure be able to use this tool.

For Python3.9 / CUDA 11.3 installation just run:
```shell
$ ./scripts/create_dpl_venv.py
```

### Note on HDF5
Some computational nodes use `h5py` module to pack data into HDF5 containers.
And we use [jpeg plugin](https://github.com/CARS-UChicago/jpegHDF5)
to save space when storing images, so you should follow their
[Installation Guide](https://github.com/CARS-UChicago/jpegHDF5#installing-the-jpeg-filter-plugin).

The best way to have appropriate versions of 
`libhdf5.so` and `libjpeg.so`
is to install [Anaconda](https://docs.anaconda.com/).

If you choose this way, you may use `scripts/build_jpeghdf5.sh` instead of provided
[build_linux](https://github.com/CARS-UChicago/jpegHDF5/blob/master/build_linux)
to compile shared library:

```shell
$ ./scripts/build_jpeghdf5.sh /path/to/anaconda /path/to/jpegHDF5/repo
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
* [ ] Continue after restart
* [ ] Better logging
* [ ] Better error handling
* [ ] Parallel over GPUs
* [ ] Nodes for visualizations
* [x] Nodes for packing datasets into containers (like H5)
* [ ] Parametrizable image format: support JPEG and PNG
* [ ] Post-check output shapes for Numpy datatypes
* [ ] Support filtering input paths
* [ ] Datatypes with multiple allowed extensions: listdir outputs files matched to ALL extensions, but that's not desirable
* [ ] Mypy and format pre-commit check
* [ ] Tests
* [ ] Dockerize
