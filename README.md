chromosome-feature-extractor
===========
<img src="https://github.com/zs-zhuang/chromosome-feature-extractor/blob/main/images/original.JPG"> <img src="https://github.com/zs-zhuang/chromosome-feature-extractor/blob/main/images/skeleton.JPG">

# Summary

This repository contains scripts that extracts chromosome features from karyotyping images such as band width, band brightness, intensity fluctuations etc. The eventual goal of extracting such features is for automated chromosome classification and abnormality detection. 

# Detailed Instruction (see bash_master) 

1. initial preparation of the original image (run log_contrast.py followed by invert.py)
2. denoise by saltpepper.py
3. convert to binary image using (ToBinary.py)
4. skeletonize the chromosome using ToSkeleton.py, get_skelston_position.py and sort_skeleton.py
5. repair the skeleton using curve_fit.py, extend_skeleton.py
6. extract chromosome features using band_shape_profile.py, shape_poly4fit.py and feature_chromosome.py

# WARNING
This repository mostly focuses on extracting chromosome features from karyotyping images. While simple machine learning algorithms such as MLP were used in an attempt to classify the different chromosomes, this is still very much a work in progress. 
