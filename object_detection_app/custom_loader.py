import torch
import numpy as np
import time
import math
from threading import Thread

from yolov9.utils.dataloaders import LOGGER
from yolov9.utils.augmentations import letterbox

"""  sources : La source du flux (par exemple, un flux vidéo).
        width et height : Dimensions du flux vidéo.
        fps : Fréquence d'images par seconde.
        frames : Nombre total d'images dans le flux.
        img_size : Dimension cible de l'image utilisée pour l'inférence (par défaut, 640).
        stride : Le pas de traitement des images dans YOLO.
        auto : Ajustement automatique des dimensions pour correspondre au modèle YOLO.
        transforms : Transformations supplémentaires optionnelles à appliquer sur les images, comme Conversion de l'espace colorimétrique : Par exemple, passer de BGR (format OpenCV) à RGB (format attendu par la plupart des modèles). """

class customLoader:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(
            self, 
            sources, 
            width,
            height,       
            fps,
            frames,
            img_size=640, stride=32, 
            auto=True, transforms=None, 
            vid_stride=1,
            ):
        

        
        #Active un mode d'optimisation dans PyTorch pour utiliser les implémentations CUDA
        #Cela permet de trouver la configuration la plus rapide pour des entrées fixes (comme les images de taille constante).
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.mode = 'stream'  #le mode d'entrée est un flux vidéo en continu
        self.img_size = img_size     
        #Un modèle avec un stride de 32 divise l'image d'entrée en cellules de 32x32 pixels pour le traitement. 
        # Chaque cellule prédit si un objet est présent ou non dans cette région.
        self.stride = stride  
        #Correspond au saut de trames lors de l'inférence sur une vidéo.
        #Si vid_stride = 1, chaque frame de la vidéo est traitée.
        #Si vid_stride = 2, une frame sur deux est analysée.
        self.vid_stride = vid_stride  # video frame-rate stride
        source = sources

        # assert isinstance(source, type(multiprocessing.Queue))
        self.sources = source
        self.imgs, self.fps, self.frames, self.threads = None, 0, 0, None
        
        # Start thread to read frames from video stream
        # st = f'feed source: {s}... '

        # s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam

        # cap = cv2.VideoCapture(s)
        # assert cap.isOpened(), f'{st}Failed to open {s}'
        w = width  #Ces dimensions sont cruciales pour redimensionner les frames et garantir que l'entrée soit compatible avec le modèle de détection.
        h = height
        fps = fps  # warning: may return 0 or nan
        #Si la source est un flux continu (comme une caméra en direct), cette variable agit comme une valeur de repli (fallback)
        # pour indiquer un flux "infini".
        self.frames = frames  # infinite stream fallback
        self.fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

        #  fps if math.isfinite(fps) else 0
        #Si le FPS récupéré est un nombre fini, il est utilisé.
        #Sinon, la valeur est mise à 0.
        #% 100
        #Si le FPS dépasse 100, il est réduit au reste de la division par 100 pour éviter des valeurs extrêmes.
        #max(..., 0)
        #Assure que le FPS ne soit jamais négatif.
        #or 30
        #Définit une valeur par défaut de 30 FPS si aucune autre valeur n'est valide. """

        _, self.imgs = source.get()  # guarantee first frame que cest fnctionnel
        #L'utilisation de threading permet de lire le flux vidéo en arrière-plan, assurant ainsi une fluidité pendant l'inférence.
        self.threads = Thread(target=self.update, args=(source, ), daemon=True)
        LOGGER.info(f" Success ({self.frames} frames {w}x{h} at {self.fps:.2f} FPS)")
        self.threads.start()
        LOGGER.info('')  # newline

        # check for common shapes
        s = np.stack([letterbox(self.imgs, img_size, stride=stride, auto=auto)[0].shape, ]) # latterbox redimmensionne pour la compatibilitee
        #Vérifie si toutes les frames du flux ont des dimensions cohérentes.
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        self.auto = auto and self.rect
        self.transforms = transforms  # optional
        if not self.rect:
            LOGGER.warning('WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, source):
        # Read stream `i` frames in daemon thread
        n, f = 0, self.frames  # frame number, frame array
        #traitement continu des trames (frames) d'un flux vidéo ou d'une source vidéo donnée
        while True and n < f:  # elle s'arrete apres avoir capturee f trames Cela permet de contrôler le nombre total de trames traitées.
            n += 1
            # cap.grab()  # .read() = .grab() followed by .retrieve()
            if n % self.vid_stride == 0:  #Lit uniquement les trames multiples de vid_stride pour réduire la fréquence de capture.
                success, im = source.get()  
                while not source.empty():
                    success, im = source.get()
                if success:
                    self.imgs = im
                else:
                    LOGGER.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                    self.imgs = np.zeros_like(self.imgs)
                    # cap.open(stream)  # re-open stream if signal was lost
            time.sleep(0.0)  # wait time Pause entre les itérations pour éviter de surcharger le processeur

    def __iter__(self):
        self.count = -1 # __next__ est appelé, self.count passe de -1 à 0,correspond au premier élément à traiter
        return self

    def __next__(self):
        self.count += 1
        # if not self.threads.is_alive() "est arrêté" or cv2.waitKey(1) == ord('q'):  # q to quit
        if not self.threads.is_alive() or self.imgs is None:
            # cv2.destroyAllWindows()
            raise StopIteration #l'itération s'arrête en levant une exception StopIteration

        im0 = [self.imgs.copy(), ]  #Copie la trame actuelle depuis self.imgs pour éviter des conflits lors de la lecture ou de la modification simultanée.
        if self.transforms:
            im = np.stack([self.transforms(im0), ])  # transforms
        else:
            im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in im0])  # redimensionner une image tout en conservant son ratio d'aspect.
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW  # ...::-1 inverse l'orde des canaux de couleurs 
            # (0,3,1,2) BHWC(Batch, Height, Width, Channels) → (Batch, Channels, Height, Width).BCHW
            im = np.ascontiguousarray(im)  # contigues

        return self.sources, im, im0, None, ''

    def __len__(self):
        return len([self.sources, ])  # 1E12 frames = 32 streams at 30 FPS for 30 years
    
""" 1 milliard de milliards de frames (1E12). Cette quantité énorme est utilisée pour illustrer la capacité du loader à 
gérer un flux de données très large ou très longue durée.
Le calcul semble indiquer qu'il est possible de traiter 32 flux vidéo à 30 images par seconde pendant 30 ans pour obtenir
 ce nombre astronomique de frames. Cela donne un sens à la gestion à long terme des données dans le loader """

