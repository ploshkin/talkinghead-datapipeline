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
    # Root to output
    --cache_dir "path/to/output/root" \
    # Execution report will be generated after run and saved here
    --report_path "path/to/report.json"
```
