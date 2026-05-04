# SVGPred
SVGPred can predict spatially variable genes (SVGs) from spatial transcriptomic data via convolutional neural network by fine-tuning the pre-trained Densenet121 model.
## Usage
### Installation
1. Clone this repo.
2. Copy the "svgpred" folder into your project directory.
### Load Spatial Transcriptomic Data
Use the 'pandas' package to read the count matrix and coordinate information. Since the purpose of count data is to generate the gene expression image/heatmap, if the count file is too large, the matrix can be split.
```
import pandas as pd
expr = pd.read_csv("path/to/your/count_df.csv",
                   header = 0,
                   index_col = 0,
                   sep = ",") # shape: (spots, genes)
coor = pd.read_csv("path/to/your/coor_df.csv",
                   header = 0,
                   index_col = 0,
                   sep = ",")
```

### generate images/heatmaps for spatial gene expressions
SVGPred provides a simple heatmap plotting function based on 'matplotlib' package.
```
from svgpred import plot
# Plot scatter with point size adjusted to fill empty space as much as possible
plot.draw_genes(expr, coor,
                save_dir = "path/to/your/imgs",
                marker = "s",
                spots_size = None) # auto-adjust if None; otherwise use the specified size
```

### Load the generated image/heatmap
Create a Dataset from the generated images.
```
from svgpred.dataset import Pred_DS
ds = Pred_DS("path/to/your/imgs")
```

### Run SVGPred
Calculate SVG scores using the fine-tuned model 
```
from svgpred.ensemble import mean_ensem
svgpred = mean_ensem(imgs_path = "path/to/your/imgs",
                     model_dir="svgpred/models/",
                     proc=10)
```

### Fine-tuning models
If you want to fine-tune the model using your own dataset, SVGPred provides a training function.
```
from svgpred.training import run_model
run_model(data_root, ds_names, savedir, proc = 4)
```
note

In '/data_root/ds_name', there should be a folder named 'img'
containing images for training and a '<ds_name>_labels.csv' file.


