# ViTSSLComparison

This project evaluates the robustness of MAE and SimCLR pre-trainied ViT's against corrupted ImageNet tiny data. The repo includes the code to pretrain ViT's from scratch using both techniques and then finetune the models on ImageNet tiny. I attempt to control for all variables besides SSL technique by using the same optimizer, epochs, etc. I end up finding that on ImageNet tiny, MAE seems to perform better across multiple severities of Gaussian Noise and Gaussian Blurring corruptions. However, SimCLR pre-training seems to perform better across multiple severities of brightness corruption:

<img width="1400" height="865" alt="image" src="https://github.com/user-attachments/assets/20ef0222-0e4f-4586-b9de-12228b26806c" />

Note that my ViT's were small due to compute constraints and ImageNet has an inadequate number of samples to train a SOTA ViT. 

One can run the whole model pipeling (pretraining and finetuning both models) using the command:
python3 -m scripts.run_pipeline --configs/pipeline.yaml 
as long as you have the ImageNet tiny dataset using the correct folder structure.

From this project, I learned the importance of pretraining vision transformer models and the potential difference in downstream task accuracy between different SSL methods for corrupted data. I also learned how to use yaml files to create model training pipelines. In a future study, I would want to compare against a ResNet, whose inductive bias should outshine ViT in this data size regime and offer more insight to the importance of model architecture for corruption robustness.
