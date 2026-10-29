## WIP
Video demo:

https://github.com/user-attachments/assets/a063bc74-9f66-4686-abb4-e586d82c540e

1 - data extraction: https://www.kaggle.com/code/rafaaaf/humminbirds-from-cub200-and-nabirds

2 - testando yolo: Peguei 1 video de passaro e 1 de beija flor. Em ambos foi encontrada a classe "bird" no video. https://pixabay.com/users/zacharycrespin-25816995/

3 - preparando os dados, src/data/build_yolo_dataset.py copia os dados de data/raw para data/yolo/images e organiza as bounding boxes em data/yolo/labels. Ocorre o split em train/val/test aqui também.

4 - fine tuning em yolo com ambos datasets, usando os bounding boxes, 100 epochs (criei modelos em weights/ e runs/detect/hummingbird_detector_yolo11s_v1/)...
https://www.kaggle.com/code/rafaaaf/beija-flor-yolo-fine-tuning2

5 - cnn

6 - Voting

7 - src/inference/ faz o que os notebooks faziam.

8 - https://zenodo.org/records/60375

9 - yt-dlp para buscar vídeos de beija flor (não deu muito certo)

10 - soft-voting

11 - testei os vídeos do VB100 e do Youtube. Salvos em runs/pipeline/kaggle_v1 (https://www.kaggle.com/code/rafaaaf/notebook12a3f1437c). Testei novamente, mas com saída estruturada com csv, runs/pipeline/kaggle_v2 (https://www.kaggle.com/code/rafaaaf/beija-flor-infer-ncia-em-v-deos-com-csv)

USE o parametro project para botar as runs/ na raiz!
track_id!!!

TO DO: testar todos os vídeos (usar o Kaggle) usando alguma métrica. Fazer Data Augmentation nas classes que não tem performance tão alta (ex: Broad-tailed_Hummingbird, principalmente nas fêmeas, Allen e Rufous, já que erram bastante).

vb100 dataset: baixados: 01 (Allen_Hummingbird), 02, 03 (Black-chinned_Hummingbird), 04 (Black_Skimmer o primeiro), 



Comandos para rodar:
```bash
# 1. Converte as anotações para o formato YOLO, faz o split e gera data/yolo e data/splits.csv
python -m src.data.build_yolo_dataset

# 2. Gera o zip com data/yolo e o detector.yaml para subir no Kaggle
python -m src.data.make_kaggle_zip

# 3. Fine-tuning do YOLO (eu treinei no Kaggle)
python -m src.detection.train

# 4. Gera os recortes das aves para a CNN
python -m src.data.make_crops

# 5. Treina a CNN de espécies
python -m src.classification.train

# 6. Avalia a CNN no conjunto de teste
python -m src.classification.evaluate

# 7. Roda detector + CNN + tracking em um vídeo
# Parâmetros opcionais, no fim do comando:
#   --name NOME         nome da pasta de saída em runs/pipeline/ (padrão: run_name do configs/inference.yaml)
#   --max-frames N      processa só os primeiros N quadros, bom para teste rápido
#   --no-show           não abre a janela do vídeo
# Exemplo: python -m src.inference.video data/videos/nome_do_arquivo.mp4 --name hummingbird_pipeline_v1 --max-frames 150
python -m src.inference.video data/videos/nome_do_arquivo.mp4
```



TESTAR IMAGEM erro_nabirds_41eb1fb7c15c48a680b631725c1ff868.png pois é roxo. ver qual a segunda espécie prevista.