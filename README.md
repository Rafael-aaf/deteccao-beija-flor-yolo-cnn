1 - data extraction: https://www.kaggle.com/code/rafaaaf/humminbirds-from-cub200-and-nabirds

2 - testando yolo: Peguei 1 video de passaro e 1 de beija flor. Em ambos foi encontrada a classe "bird" no video. https://pixabay.com/users/zacharycrespin-25816995/

3 - preparando os dados, src/data/build_yolo_dataset.py copia os dados de data/raw para data/yolo/images e organiza as bounding boxes em data/yolo/labels. Ocorre o split em train/val/test aqui também.

4 - fine tuning em yolo com ambos datasets, usando os bounding boxes, 100 epochs (criei modelos em weights/ e runs/detect/hummingbird_detector_yolo11s_v1/)...
https://www.kaggle.com/code/rafaaaf/beija-flor-yolo-fine-tuning2

5 - cnn

6 - Voting

7 - src/inference/ faz o que os notebooks faziam.

USE o parametro project para botar as runs/ na raiz!

TO DO: colocar como output 2 espécies, testar todos os vídeos (usar o Kaggle) usando alguma métrica. Fazer Data Augmentation nas classes que não tem performance tão alta (ex: Broad-tailed_Hummingbird, principalmente nas fêmeas, Allen e Rufous, já que erram bastante).

vb100 dataset: baixados: 01 (Allen_Hummingbird), 02, 03 (Black-chinned_Hummingbird), 04 (Black_Skimmer o primeiro), 