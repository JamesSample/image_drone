# Image Drone

Applying machine learning techniques for automatic classification of drone images.

## DRONING project

 1. **[Exploratory analysis](https://nbviewer.jupyter.org/github/JamesSample/image_drone/blob/master/notebooks/drone_ml_image_class.ipynb)**. A initial attempt at creating a training dataset and applying a simple Random Forest classifier.
 
 2. **[Refining the classification scheme](http://nbviewer.jupyter.org/github/JamesSample/image_drone/blob/master/notebooks/drone_ml_janne.ipynb)**. Using mosiaced images and a more comprehensive training dataset to see how much detail can be reasonably extracted using a simple random forest algorithm.

## Frisk Oslofjord project

 1. **[Initial raster processing](https://nbviewer.jupyter.org/github/JamesSample/image_drone/blob/master/notebooks/frisk_oslofjord_raster_proc.ipynb)**. Combining the RGB and multispectral datasets into a single 8-band image.
 
 2. **[Image classification](https://nbviewer.jupyter.org/github/JamesSample/image_drone/blob/master/notebooks/frisk_oslofjord_ml.ipynb)**. Training and evaluating a Random Forest model, then applying it to predict substrate classes for the full Akerøya dataset.
 
 2. **[Tuning and comparing different models](https://nbviewer.jupyter.org/github/JamesSample/image_drone/blob/master/notebooks/frisk_oslofjord_mod_comp.ipynb)**. Initial code for calibrating and comparing a range of supervsied classification algorithms.
 
 <p align="center">
  <img src="/images/data_processing_workflow.png" alt="Frisk Oslofjord workflow" width="800" />
</p>
