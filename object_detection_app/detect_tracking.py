import os
import sys
from pathlib import Path
# Ajoutez ces importations en haut de votre fichier
from collections import defaultdict
import multiprocessing

import torch
yolo_root = Path(__file__).resolve().parent / "yolov9"
sys.path.append(str(yolo_root))

from yolov9.models.common import DetectMultiBackend
from yolov9.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov9.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov9.utils.torch_utils import select_device, smart_inference_mode


from pathlib import Path
import math
import torch
import numpy as np

from yolov9.deep_sort_pytorch.utils.parser import get_config
from yolov9.deep_sort_pytorch.deep_sort import DeepSort

from yolov9.models.common import DetectMultiBackend
from yolov9.utils.dataloaders import IMG_FORMATS, VID_FORMATS
from yolov9.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)

from yolov9.utils.torch_utils import select_device, smart_inference_mode
from custom_loader import customLoader





def initialize_deepsort():
    # Create the Deep SORT configuration object and load settings from the YAML file
    cfg_deep = get_config()
    cfg_deep.merge_from_file(yolo_root / "deep_sort_pytorch/configs/deep_sort.yaml")
    # Initialize the DeepSort tracker
    deepsort = DeepSort(yolo_root / cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                        # min_confidence  parameter sets the minimum tracking confidence required for an object detection to be considered in the tracking process
                        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        #nms_max_overlap specifies the maximum allowed overlap between bounding boxes during non-maximum suppression (NMS)
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        #max_iou_distance parameter defines the maximum intersection-over-union (IoU) distance between object detections
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        # Max_age: If an object's tracking ID is lost (i.e., the object is no longer detected), this parameter determines how many frames the tracker should wait before assigning a new id
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                        #nn_budget: It sets the budget for the nearest-neighbor search.
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True
        )

    return deepsort

def xyxy_to_xywh(x):
    # Convert [x1, y1, x2, y2] to [center_x, center_y, width, height]
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # center x
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # center y
    y[:, 2] = x[:, 2] - x[:, 0]        # width
    y[:, 3] = x[:, 3] - x[:, 1]        # height
    return y

@smart_inference_mode()
def start_tracker(
        # weights=yolo_root / 'best.pt',  # model path or triton URL
        source:multiprocessing.Queue,
        drawings:multiprocessing.Queue,
        fps,
        data=yolo_root / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 480),  # inference size (height, width)
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
):
    
    # Initialize Deep SORT
    deepsort = initialize_deepsort()

    # Initialisez un dictionnaire pour stocker les positions et vitesses
    object_paths = defaultdict(list)  # Dictionnaire pour stocker les chemins
    object_speeds = {}  # Dictionnaire pour stocker les vitesses

    # for path, im, im0s, vid_cap, s in dataset:
    while True:
        print('predicting')

        detections = []
        detection, im0 = source.get()
        detections.append(detection)

        print('doing prediction')

        # Update the tracker with the current detections
        if len(detections) > 0:
            detections = np.array(detections)
            bbox_xywh = xyxy_to_xywh(detections[:, :4])  # Convert to xywh format
            confidences = detections[:, 4]  # Get confidence scores
            classes = detections[:, 5].astype(int)  # Get classes

            # Update Deep SORT tracker
            outputs = deepsort.update(bbox_xywh, confidences, classes, im0)
            
            print(f"Detections: {detections}")  # Voir si des détections sont détectées
            print(f"Outputs (DeepSORT): {outputs}")  # Voir si le tracker génère des sorties
            

            # Draw tracking results
            if outputs is not None and len(outputs) > 0:
                for output, cls, conf in zip(outputs, classes, confidences):
                    x1, y1, x2, y2, track_id = output[:5]
                    # Calculer la position actuelle
                    current_position = (x1 + x2) / 2, (y1 + y2) / 2  # Centre de la boîte

                    # Stocker la position pour dessiner le chemin
                    object_paths[track_id].append(current_position)

                    # Calculer la vitesse si nous avons une position précédente
                    if len(object_paths[track_id]) > 1:
                        previous_position = object_paths[track_id][-2]
                        print(f"Previous position: {previous_position}, Current position: {current_position}")
                        if previous_position is None:
                            previous_position = current_position
                            continue  # Passez à la prochaine itération pour éviter un calcul erroné
                        if previous_position and current_position:
                            distance = math.sqrt((current_position[0] - previous_position[0]) ** 2 +
                                                (current_position[1] - previous_position[1]) ** 2)
                        else:
                            distance = 0  # Pas de mouvement détecté
                        
                        # Affichez la distance pour le débogage
                        print(f"Distance: {distance} pixels")
                        # Calculer la distance en millimètres
                        #distance_millimeters = distance * 1000  # Convertir en mm
                         # Calculer la distance en millimètres
                        pixel_to_meter = 100.0  # Ajustez cette valeur en fonction de votre échelle
                        distance_millimeters = (distance / pixel_to_meter) * 1000  # Convertir en mm
                        print(f"Distance (mm): {distance_millimeters}")
                        # Affichez la distance pour le débogage
                        print(f"Distance: {distance_millimeters} pixels")
                        # Supposons que le temps entre les images est constant (par exemple, 1/30 seconde pour 30 FPS)
                        # Assurez-vous que vous obtenez la fréquence d'images correcte
                        fps = fps  # Utilisez la fréquence d'images de la vidéo
                        print(f"FPS: {fps}")
                        time_interval = 1 / fps  # Temps entre les images
                        #time_interval = 1 / 30  # Ajustez en fonction de votre fréquence d'images
                        speed = distance_millimeters / time_interval
                        object_speeds[track_id] = speed
                        # Affichez la vitesse pour le débogage
                        print(f"Speed: {speed:.2f} mm/s")
                    else:
                        # Default speed if no previous position exists
                        object_speeds[track_id] = 0

                    # Dessiner le chemin
                    if len(object_paths[track_id]) > 1:
                        for i in range(1, len(object_paths[track_id])):
                            drawings.put(('line', [int(object_paths[track_id][i-1][0]), int(object_paths[track_id][i-1][1]), int(object_paths[track_id][i][0]), int(object_paths[track_id][i][1])]))
                    # Créez le label avec la classe, la confiance et la vitesse
                    label = f'ID: {track_id}, Speed: {object_speeds.get(track_id, 0):.2f} mm/s'

                    drawings.put(('points', [x1, y1, x2, y2, label]))
                    # Draw object center
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)


            # Gestion de la prédiction en cas de perte temporaire
            for track_id in object_paths.keys():
                if track_id not in [output[4] for output in outputs]:  # Si l'ID n'est pas dans les résultats actuels
                    if len(object_paths[track_id]) > 1 and track_id in object_speeds:  # Si une trajectoire existe
                        last_position = object_paths[track_id][-1]
                        #if track_id in object_speeds:
                        # Prédire la position future
                        predicted_position = (
                            last_position[0] + object_speeds[track_id] * time_interval,
                            last_position[1] + object_speeds[track_id] * time_interval
                        )
                        # Dessiner une ligne depuis la dernière position
                        drawings.put(('line', [last_position[0], last_position[1], predicted_position[0], predicted_position[1]]))
                        # Mettre à jour la trajectoire avec la position prédite
                        object_paths[track_id].append(predicted_position)
        

