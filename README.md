# SVGPred
SVGPred can predict spatially variable genes (SVGs) from spatial transcriptomic data via convolutional neural network by fine-tuning the pre-trained ResNet50 model.
## Usage
### Load Spatial Transcriptomic Data
Use the pandas package to read the count matrix and coordinate information. Since the purpose of count data is to generate the gene expression image/heatmap, if the count file is too large, the matrix can be split.
```
import pandas as pd
expr = pd.read_csv("Dataset/count_df.csv",
                   header = 0,
                   index_col = 0,
                   sep = ",")
coor = pd.read_csv("Dataset1/coor_df.csv",
                   header = 0,
                   index_col = 0,
                   sep = ",")
```

### generate images/heatmaps for spatial gene expressions
SVGPred provides a simple heatmap plotting function based on 'matplotlib'. package.
```
from svgpred import plot
plot.draw_genes(expr, coor, save_dir = "Dataset/imgs")
```

### Load the generated image/heatmap
Create a Dataset from the generated images.
```
from svgpred.dataset import Pred_DS
ds = Pred_DS("Dataset/imgs")
```

### Run SVGPred
Calculate SVG scores using the fine-tuned model 
```
from svgpred.ensemble import mean_ensem
svgpred = mean_ensem()
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
