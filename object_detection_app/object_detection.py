import multiprocessing
from pathlib import Path
import sys
import time

import torch


yolo_root = Path(__file__).resolve().parent / "yolov9"
sys.path.append(str(yolo_root))

from yolov9.utils.general import check_imshow, check_img_size, Profile, LOGGER, non_max_suppression, scale_boxes
from yolov9.models.common import DetectMultiBackend
from yolov9.utils.torch_utils import select_device

# testing modifying LoadStreams class
from custom_loader import customLoader


""" source: multiprocessing.Queue : Une file de messages utilisée pour échanger des informations entre le processus principal 
            (le parent) et les processus enfants. Cela permet de lire en continu les frames vidéo à traiter.
centers_queue: multiprocessing.Queue: Une file de messages utilisée pour envoyer les coordonnées du centre
            des objets détectés ainsi que les informations de détection 
            
iou_thres :si deux détections se chevauchent avec une intersection sur l'union supérieure à iou_thres, 
            seule la détection avec la plus grande confiance (conf) est gardée"""

def yolo_detector(
        source: multiprocessing.Queue,
        centers_queue: multiprocessing.Queue,
        width,
        height,
        fps,
        frames,
        imgsz=(640, 480),
        weights='best.pt',
        data=yolo_root / 'data/coco.yaml',
        vid_stride=1,
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold  Lorsqu'on diminue le seuil (iou_thres) : Cela permet de conserver plus de détections [intersection over union]
        max_det=5,  # maximum detections per image
        # line_thickness=3,  # bounding box thickness (pixels)
        ):

    device = select_device('cpu')
    weights = yolo_root / weights
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False) #Instanciation du modèle YOLO
    #fp16= None :indique qu'on ne veut pas utiliser la précision flottante 16 bits pour l'inférence pour des raisons de performances ou de compatibilité.
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size pour ne pas avoir des erreurs d'incompatibilitee

    view_img = check_imshow(warn=True)  #vérifier si une interface d'affichage est disponible
    if not view_img:
        print('ERROR: environment unsuitable')
        exit()
    
    dataset = customLoader(source, width, height, fps, frames, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    # dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = len(dataset) #combien d'images le modèle traite à la fois.


    # Run inference
    #warmup est une étape critique avant l'inférence réelle, utilisée pour préparer le modèle à traiter les images.
    #Si pt est vrai ou si model.triton est actif, cela signifie que le modèle est optimisé pour Triton
    # (une plateforme d'accélération d'inférence). Dans ce cas, il utilise un batch size de 1.
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  #3 est le nombre de canaux
    #*imgsz : Il décompresse les dimensions de l'image (h, w) fournies dans imgsz pour les passer comme arguments distincts.
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    #dt ..... Un tuple contenant trois objets Profile(). Ces objets sont utilisés pour mesurer les temps de traitement
    for path, im, im0s, vid_cap, s in dataset:
        # Charger l'image et effectuer les transformations nécessaires
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)  # Convertir l'image from numpy en tensor PyTorch Cela garantit que l'image 
            #est utilisée sur le même appareil que le modèle pour éviter les transferts inutiles de données
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32   format de flottant en demi-précision (FP16)
            im /= 255  # Normaliser les valeurs de pixel de l'échelle 0 - 255 à 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]   # Étendre pour ajouter la dimension batch
            """ Vérifie si l'image n'a qu'une seule dimension (H x W x C). Si c'est le cas, ajoute une dimension supplémentaire à 
            l'image pour le batch (im[None]) """

        # Profiling de l'inférence YOLO
        with dt[1]:
            pred = model(im, augment=False)  #désactive l'augmentation pour éviter de modifier les données d'entrée.

        # NMS Profiling de l'annotation et de la visualisation
        with dt[2]:
            pred = pred[0][1] if isinstance(pred[0], list) else pred[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)
            #réduire le nombre de détections en supprimant les détections redondantes 
            #(boîtes de détection qui se chevauchent avec des scores de confiance similaires)
        
        # print(f'{len(pred)} predictions')
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            im0, frame = im0s[i].copy(), dataset.count  #im0 est utilisée pour dessiner des boîtes de détection et autres visualisations sans altérer l'image d'origine.
            s += f'{i}: ' #suivre quelle image est actuellement traitée.
           
            s += '%gx%g ' % im.shape[2:] #est un formatage de chaîne qui insère ces dimensions dans la chaîne s, indiquant les dimensions de l'image actuelle.
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #normaliser les bbow lors de la conversion des dimensions d'une image à une autre
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # elles sont arrondies aux entier
                #les premières quatre colonnes de la matrice det. Dans le contexte des résultats de la détection d'objets,
                # ces quatre valeurs correspondent aux coordonnées des boîtes englobantes (bounding boxes) dans l'image.

                # Print results
                for c in det[:, 5].unique(): #Pour chaque classe unique détectée dans les prédictions.
                    n = (det[:, 5] == c).sum()  # Calcule le nombre de détections pour cette classe.
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # Ajoute une description des détections pour chaque classe à la chaîne s
                    #Si n est supérieur à 1, alors 's' est ajouté à la chaîne, sinon, rien n'est ajouté.

                # Write results
                #Boucle sur chaque détection et affiche les informations de détection
                for *xyxy, conf, cls in reversed(det):   #Parcourt les détections de manière inversée pour traiter les détections par confiance décroissant.
                    # Convertir les coordonnées en pixels entiers
                    bbox_pixels = tuple(map(int, xyxy))  # (x1, y1, x2, y2)

                    # Calcul du centre
                    x1, y1, x2, y2 = bbox_pixels
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Afficher les résultats
                    print(f"Bounding box: {bbox_pixels}")
                    print(f"Object center: ({center_x}, {center_y})")

                    centers_queue.put(((center_x, center_y), bbox_pixels, conf))

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        time.sleep(0.1)
    
    print('YOLO detection shut down')
    return






