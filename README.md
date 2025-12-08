# Super Resolution

## Prerequisite
The packages needed are listed in `pyproject.toml`. You can use [uv](https://github.com/astral-sh/uv) and execute the command below

```sh
uv sync
```

to install the packages. Afterward activate the virtual environment created.

## Training args

SRResNet
```sh
python train_resnet.py --iterations 100000
```

SRGAN
```sh
python train_srgan.py --iterations 100000 --warmup_model ./checkpoints/srresnet_66.pth.tar --warmup_model ./checkpoints/srresnet_66.pth.tar
```

SRRaNet
```sh
python train_srgan.py --iterations 100000 --exclude_activation --loss_strategy relativistic  --exclude_bn --psnr_mode
```

SRRaGAN
```sh
python train_srgan.py --iterations 200000 --exclude_activation --loss_strategy relativistic --exclude_bn --checkpoint ./checkpoints/srragan_psnr_nobn_66.pth.tar
```

## Running the front-end application

Ensure that the models are available in `frontend/public/models/`. If they aren't, follow the steps below:


1. Export the models into `.onnx`
```sh
python export_onnx.py ./checkpoints/srresnet_66.pth.tar ./srresnet.onnx

python export_onnx.py ./checkpoints/srragan_psnr_nobn_66.pth.tar ./srranet.onnx

python export_onnx.py ./checkpoints/srgan_66.pth.tar ./srgan.onnx

python export_onnx.py ./checkpoints/srragan_nobn_132.pth.tar ./srragan.onnx
```
2. Copy to `frontend/public/models/`
```sh
cp ./srresnet.onnx ./srranet.onnx ./srgan.onnx ./srragan.onnx frontend/public/models/
```

Afterwards you can run the commands below to run the application locally.
```sh
cd frontend/
npm install
npm run dev
```

## Utilities
There are multiple utility scripts provided in this repository.

- `create_data_lists.py`: Creates a `.json` file pointing to the training and testing files. This must be run before training is executed.
- `export_onnx.py`: Exports the model into a `.onnx` file for the front-end application.
- `eval.py`: Run evaluation of the model on test files. Requires `create_data_lists.py`. 
- `extract_loss.py`: Extract the losses stored in the model checkpoints.
- `get_bicubic.py`: Retrieves a bicubic downsampled image to run inference on.
- `demonstration.py`: Run a comparison between SRResNet and SRGAN (deprecated).
- `inference.py`: Run an inference on a single model.

