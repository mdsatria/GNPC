# GNPC (Graph for NPC Prognosis)

A Python project for prognosis prediction on Whole Slide Images (WSIs) using Graph Neural Networks (GNNs) and PyTorch Geometric.

This code is from the publication **Multimodal AI-Based Risk Stratification for Distant Metastasis in Nasopharyngeal Carcinoma**. It provides an end-to-end pipeline for preprocessing WSIs, extracting patch embeddings and morphology features, constructing graphs, and training deep learning models for prognosis tasks.


## Features

✅ WSI preprocessing  
✅ Graph construction (patch-level, region-level)  
✅ Prognosis model with PyTorch Geometric  
✅ Training, evaluation, and visualisation scripts

## Installation

### Requirements

- Python >= 3.10  
- PyTorch >= 2.0  
- torch-geometric  
- OpenSlide  
- CUDA >= 12.1

### Install dependencies

1. **Create conda environment**
```bash
conda create --name gnpc_test python=3.10

conda activate gnpc_test
```

2. **Install PyTorch and torch-geometric**
```bash
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

pip install torch_geometric -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

3. **Install packages from requirements.txt**
```bash
pip install -r requirements.txt
```

4. **Clone the repository**
```bash
git clone https://github.com/mdsatria/GNPC
cd GNPC
chmod +x full_pipeline.sh make_embedding.sh make_graph.sh make_morphology.sh make_patch.sh make_train.sh
```

## Usage

The entire pipeline can be run in one go by executing `full_pipeline.sh`.  
Our pipeline consists of:  
(1) Patch extraction,  
(2) Patch embedding,  
(3) Extraction of patch-based cell morphology features,  
(4) Graph construction, and  
(5) Model training.

Each step can also be run individually by executing the following shell scripts:

1. **Patch extraction**  
   We generate patches of size 512x512 at 20X magnification (0.5 MPP). This can be done by executing `make_patch.sh`. All WSIs should be listed in a CSV file following the format shown in `sample_data/csv/dataset.csv`.

   Below are the patch extraction parameters `make_patch.sh`:
   ```bash
   csv = path to the CSV file listing the slides
   dir_output = directory to store the patches
   downsample = parameter for WSI image segmentation
   kernel_close = parameter for WSI image segmentation
   threshold = parameter for WSI image segmentation
   med_blur = parameter for WSI image segmentation
   max_white_area = parameter for WSI image segmentation
   white_threshold = parameter for WSI image segmentation
   patch_size = patch size
   target_mag = patch resolution/magnification
   percent = minimum percentage of tissue to be considered foreground
   save_thumb = save the thumbnail
   save_mask = save WSI mask
   save_coord = save coordinates of patches
   save_img = save image patches
   stitch = stitch image patches and save
   ```

2. **Patch embedding**  
   This step extracts deep features from the image patches generated in the previous step. Users can change the model used to extract the features. For foundation models, please refer to each publication for details on the feature extraction process.
   
   Below are the patch embedding parameters `make_embedding.sh`:
   ```bash
   cohort = cohort name of the dataset
   dir_wsi = directory where WSIs are stored
   dir_patch = directory of patch coordinates
   csv_wsi = path to WSIs dataset
   dir_save = directory to store the embedding output
   models = model name for feature extraction
   ```

3. **Morphology feature extraction**  
   This step is optional. For this step, all nuclei in the WSIs need to be detected and segmented using a model like [HoverNet](https://github.com/vqdang/hover_net), and stored in a pickle file following the schema provided in `sample_data/nuclei_prediction/nuclei_sample.pkl`.

   Below are the morphology feature extraction parameters `make_morphology.sh`:
   ```bash
   csv_slide = path to the CSV file listing the slides
   dir_wsi = directory where WSIs are stored
   dir_nuclei = directory containing nuclei prediction pickle files
   dir_patch = directory of patch coordinates
   dir_save = directory to store the extracted morphology features
   log_file = path to save the log file
   target_mpp = target resolution in microns per pixel (MPP) for scaling
   ```


4. **Graph construction**  
   This step constructs graphs from the extracted features, embedding data, and (optionally) morphology features. The graphs are built at the patch or region level and can incorporate both embedding and nuclei information depending on the configuration.

   Below are the graph construction parameters `make_graph.sh`:
   ```bash
   csv = path to the CSV file listing the slides
   dir_emb = directory containing patch embedding files
   dir_patch = directory of patch coordinates
   dir_nuc = directory of nuclei morphology feature files (optional, required if use_nuc is enabled)
   dir_save = directory to store the constructed graph files
   cohort = name of the cohort or dataset
   weighted = flag to indicate whether to construct weighted graphs 
   use_nuc = flag to indicate whether to include nuclei morphology features 
   emb = name of the embedding model used (for tracking or specific processing)
   max_dist = maximum distance threshold (in microns or pixels) for connecting nodes in the graph
   ```

5. **Model training (GNN)**  
   This step trains a Graph Neural Network (GNN) model for prognosis prediction using the constructed graph data. The training process is configured through a JSON file that specifies the model architecture, hyperparameters, dataset paths, and other training settings.

   Below is the model training parameter:
   ```bash
   config_json = path to the JSON configuration file containing all training settings, including model parameters, data paths, and hyperparameters
   ```


## Citation

If you use this code or find it helpful in your research, please cite the following publication:

```

@article{zhou_multimodal_2025,
	title    = {Multimodal AI-Based Risk Stratification for Distant Metastasis in Nasopharyngeal Carcinoma},
   author   = {Zhou, Jiayu and Wibawa, Made Satria and Wang, Ruoyu and Deng, Ying and Huang, Haoyang and Luo, Zhuoying and Xia, Yue and Guo, Xiang and Young, Lawrence S. and Lo, Kwok-Wai and Rajpoot, Nasir and Lv, Xing},
   year     = {2025},
	url      = {https://www.medrxiv.org/content/10.1101/2025.01.28.25321109v1},
	doi      = {10.1101/2025.01.28.25321109},
}

```
