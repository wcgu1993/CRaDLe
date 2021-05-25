# CRaDLe: Deep Code Retrieval Based on Semantic Dependency Learning

PyTorch implementation of [CRaDLe](https://reader.elsevier.com/reader/sd/pii/S0893608021001568?token=9D9A8D8C785F194BEAC362C63CC2B65CCCC69293124E0BC360103DAAFBBF43DC86EB38936D6045AC736FE10A5E564700&originRegion=us-east-1&originCreation=20210525121434). This model is modified from the project shared by [Deep Code Search](https://github.com/guxd/deep-code-search).
## Dependency
> Tested in MacOS 10.12, Ubuntu 16.04
* Python 3.6
* PyTorch 
* tqdm

 ```
 pip install -r requirements.txt
 ```
 

## Code Structures

 - `models`: neural network models for code/desc representation and similarity measure.
 - `modules.py`: basic modules for model construction.
 - `train.py`: train and validate code/desc representaton models; 
 - `configs.py`: configurations for models defined in the `models` folder. 
   Each function defines the hyper-parameters for the corresponding model.
 - `data_loader.py`: A PyTorch dataset loader.
 - `utils.py`: utilities for models and training. 


## Usage

   ### Data Preparation
  To train and test our model:
  
  1) Download and unzip real dataset from [Google Drive](https://drive.google.com/drive/folders/1QtQPq9clBafqCcp80GwoAau8DlUqLeor?usp=sharing).
  
  2) Put all the data files into the `/data/github` folder . 
  
   ### Configuration
   Edit hyper-parameters and settings in `config.py`

   ### Train
   
   ```bash
   python train.py --model JointEmbeder
   ```
   

## Citation

 If you find it useful and would like to cite it, the following would be appropriate:
```
@article{gu2020cradle,
  title={CRaDLe: Deep Code Retrieval Based on Semantic Dependency Learning},
  author={Wenchao Gu and Zongjie Li and Cuiyun Gao and Chaozheng Wang and Hongyu Zhang and Zenglin Xu and Michael R. Lyu},
  journal={Neural Networks},
  year={2021}
}
```
 (2021). CRaDLe: Deep code retrieval based on semantic Dependency Learning. Neural Networks.