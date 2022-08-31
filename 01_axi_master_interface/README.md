# Vivado Accelerator Backend w/ AXI-master Interface

This is a workspace for testing the integration of the Vivado Accelerator backend with AXI-master interface. It is a work-in-progress repository, but it should help us to converge on a PR for the official hls4ml repository.

The _hls4ml_ fork and branch that we use in this workspace is
- https://github.com/fnal-fastml/hls4ml/tree/vivado-accelerator-axi-master-interface

```
conda env create -f environment.yml
conda activate hls4ml-vivado-accelerator
pip install qkeras==0.9.0
pip uninstall hls4ml
pip install git+https://github.com/fnal-fastml/hls4ml.git@vivado-accelerator-axi-master-interface#egg=hls4ml[profiling]
```

## Profile the Model
```
make run-profile
```

## Run HLS
```
make run
```
