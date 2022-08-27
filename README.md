# BIFI para CodeNeyPy
Proyecto de Deep Learning

## Pre-requisitos
- CUDA 10.1 para arriba
- conda - Miniconda 3.7 - Docs: https://docs.conda.io/en/latest/miniconda.html - Instalar según la plataforma
### Opcionales:
- Por lo menos 1 NVIDIA GPU como mínimo (tanto el entrenamiento como inferencia son procesos EXTREMADAMENTE lentos con CPU) - para el proyecto se uso una instancia en Google Cloud Platform con una NVIDIA T4 de 16GB
- Drivers de NVIDIA instalados de forma correcta

## Instalación
1. Hacer clon del repo: https://github.com/michiyasunaga/BIFI
2. cd BIFI/utils
3. Borrar el directorio fairseq con todos sus contenidos
4. Clonar fairseq con git clone git clone --depth 1 --branch v0.9.0 https://github.com/facebookresearch/fairseq
5. Para realizar tanto el entrenamiento como la inferencia de forma existosa, es necesario que se utilice esta versión de fairseq compatible con pytorch 1.4.0 y torchvision 0.5.0
6. cd fairseq/fairseq/tasks/ y en la línea 150 del archivo fairseq_task.py, modificar lo siguiente:
                indices, dataset, max_positions, raise_exception=(not ignore_invalid_inputs),
                indices, dataset, max_positions, raise_exception=False,
7. En la línea 93 del archivo translation.py, agregar (con correcta identación) lo siguiente:
tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
8. Modificar la línea 96 de este mismo archivo
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,

9. cd ../../../..
10. conda create -n BIFI python=3.7.7
11. conda activate BIFI
12. pip install tqdm
13. pip install torch==1.4.0 torchvision==0.5.0
14. cd utils/fairseq
15. pip install -e .
16. pip install numpy==1.20.1 editdistance
17. cd ../..

### Instrucciones para realizar inferencia sobre DrRepair pre-entrenado utilizando BIFI
18. Una vez configurado el ambiente, descargar el modelo del fixer de BIFI pre-entrenado después de dos rondas (el definitivo)
19. Descargar, además, el dataset de CodeNetPy utilizado en este proyecto (https://www.kaggle.com/datasets/alexjercan/codenetpy). Este dataset fue curado desde CodeNet, el dataset de benchmarking desarrollado por IBM en varios lenguajes de programación, incluyendo Python. Fuente: https://github.com/IBM/Project_CodeNet
20. mkdir preprocess/inference/model-fixer
21. Copiar el modelo descargado en el directorio anterior
22. mkdir datasets
23. Copiar los contenidos del dataset de CodeNetPy (descripciones de problemas, codenetpy_train.json y codenetpy_test.json) en este directorio anterior
24. Correr el notebook eda.ipynb para el análisis exploratorio de datos
25. Correr el notebook preprocessing.ipynb para el preprocesamiento del dataset
26. export PYTHONPATH=.
27. python src/run_fixer_inference.py --round_name inference --gpu_ids '0'
28. python src/evaluate_fixer_inference.py  --round_name inference

## Hardware y software usado en este proyecto
- 1 x NVIDIA Tesla T4 de 16GB
- Driver Version: 510.47.03
- CUDA Version: 11.6
- SSD de boot de 50GB
- SSD para alojar el proyecto de 450GB
- Google, Debian 10 based Deep Learning VM with CUDA 11.0, M95, Base CUDA 11.0, Deep Learning VM Image with CUDA 11.0 preinstalled.