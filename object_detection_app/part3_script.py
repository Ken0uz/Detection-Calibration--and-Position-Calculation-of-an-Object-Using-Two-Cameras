import cv2
import multiprocessing
import numpy as np
import math
# from time import sleep
import streamlit as st
# import detection_app
from object_detection import yolo_detector
from detect_tracking import start_tracker

print('..')

class MyApp:

    def __init__(self, right_camera, right_camera_id, left_camera, left_camera_id, calib, constant_calib, detect, method, model_path, conf_thresh, max_det, device, camera_dist, predict, n_row=7, n_col=9, num_calib_im=6):

        # running parameters 
        self.right_camera = right_camera
        self.left_camera = left_camera
        self.calib = calib
        self.constant_calib = constant_calib
        self.detect = detect
        self.method = method
        # yolo params
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.max_det = max_det
        self.device = device
        # depth calc param
        self.camera_dist = camera_dist
        # prediction
        self.predict = predict


        self.n_row = n_row
        self.n_col = n_col
        self.baseline = camera_dist

        # capture = cv2.VideoCapture(self.cam_id, cv2.CAP_V4L2)
        width = 640
        height = 480

        print('initiating cameras')
        # right camera variables (vR n'est pas utilises)
        self.cap_R = cv2.VideoCapture(right_camera_id)
        # self.cap_R.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap_R.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap_R.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # left camera variables
        self.cap_L = cv2.VideoCapture(left_camera_id)
        # self.cap_L.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap_L.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap_L.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        print('cameras initiated')

        # common variables
        self.uR, self.vR = None, None
        self.uL, self.vL = None, None

        self.focal_length_x, self.cx, self.cy = None, None, None
        self.centers_right, self.centers_left = None, None
        self.origin_coords_right, self.origin_coords_left = None, None # coordinates of the real world basis origin point (first point of checkered chessboard) in the camera
        self.origin_cam_basis = None # coordinates of the real world basis origin point (first point of checkered chessboard) with camera as basis
        self.cam_mid_point = None # coordinates of the camera world basis origin point (mid point) with real world as basis
        self.X_cam, self.Y_cam, self.Z_cam = None, None, None
        self.X_real, self.Y_real, self.Z_real = None, None, None

        # distance of object from mid point between cameras
        self.distance_cam = None  # Distance entre les caméras en mm
        self.distance_checker = None  # Distance entre les caméras en mm
        # number of images used when calibrating in each of the two cameras
        self.num_calib_im = num_calib_im

        # serving images for calibration
        self.image_queue_R = multiprocessing.Queue()
        self.image_queue_L = multiprocessing.Queue()
        # getting results of calibration
        self.result_queue_R = multiprocessing.Queue()
        self.result_queue_L = multiprocessing.Queue()
        # getting results of calibration (checkboard corners)
        self.corners_queue_R = multiprocessing.Queue()
        self.corners_queue_L = multiprocessing.Queue()
        # serving images for yolo detection
        self.video_feed_R = multiprocessing.Queue()
        self.video_feed_L = multiprocessing.Queue()
        # getting results of object detection (centers, bounding box)
        self.centers_queue_R = multiprocessing.Queue()
        self.centers_queue_L = multiprocessing.Queue()
        # sending object detections for prediction
        self.detections_queue_R = multiprocessing.Queue()
        self.detections_queue_L = multiprocessing.Queue()
        # getting predictions
        self.predictions_queue_R = multiprocessing.Queue()
        self.predictions_queue_L = multiprocessing.Queue()

        # for prediction (storing made tracks)
        # self.object_paths = {}

        # 
        self.compute3d = None



    # wrapper for object detection
    def detect_objects(video_feed, centers_queue, w, h, fps, frames, method, model_path, conf_thresh, max_det, device):
        """
        Wrapper for detection method

        Args:
        - frame : Image à analyser.
        - lower_red1, upper_red1 : Plage HSV pour le rouge vif.
        - lower_red2, upper_red2 : Plage HSV pour le rouge somb
    Returns:
        - frame : Image annotée avec des rectangles et points.
        - centers : Liste des centres des objets rouges détectés.
        """
        if method == 'yolo':
            yolo_detector(video_feed, centers_queue, w, h, fps, frames, weights=model_path, conf_thres=conf_thresh, max_det=max_det)
        elif method == 'color_detector':
            MyApp.detect_red_objects(video_feed, centers_queue)

    # wrapper for object prediction
    def predict_objects(detections_queue, predictions_queue, fps):

        start_tracker(detections_queue, predictions_queue, fps)

    # Calcul de la position 3D d'un objet
    def cam_2D_to_cam_3D(self, uL, vL, uR, vR=0):
        """
        Calcule la position 3D (X, Y, Z) d'un point en utilisant la stéréovision.

        Args:
        - uL, uR : Coordonnées X du point dans les images de gauche et de droite.
        - vL : Coordonnée Y dans l'image de gauche.
        - focal_length_x : Distance focale de la caméra.
        - baseline : Distance entre les deux caméras (en millimètres).
        - cx, cy : Coordonnées du centre optique.

        Returns:
        - X, Y, Z : Coordonnées du point dans l'espace 3D.
        """
        disparity = uL - uR  # Différence entre les positions du point dans les deux images
        if disparity == 0:
            raise ValueError("La disparité ne peut pas être nulle.")  # Empêche la division par zéro
        Z = (self.baseline * self.focal_length_x) / disparity  # Profondeur (Z)
        X = Z * (uL - self.cx) / self.focal_length_x  # Position X
        Y = Z * (vL - self.cy) / self.focal_length_x  # Position Y
        return (int(X), int(Y), int(Z))
    
    def cam_3D_to_real_3D(self, x, y, z):
        if self.origin_cam_basis is not None:
            return x - self.origin_cam_basis[0], y - self.origin_cam_basis[1], z - self.origin_cam_basis[2]
        
        return None, None, None

    def get_calibration_error(ret, mtx, dist, rvecs, tvecs, imgpoints, objpoints):
        # Reprojection Error
        mean_error = 0

        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error

        print("----------total error: {}".format(mean_error/len(objpoints)) )

    def calibrate_camera(images, corners_queue, n_row=7, n_col=9):
        """core camera calibration function."""
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ...
        
        objp = np.zeros((n_row*n_col,3), np.float32)
        objp[:,:2] = np.mgrid[0:n_row,0:n_col].T.reshape(-1,2)

        objpoints = []  # 3D points in real-world space
        imgpoints = []  # 2D points in image plane
        imgAugmnt = None

        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (n_row, n_col), None)
            if ret:
                corners_queue.put((ret, corners))  # send corners to draw them on-screen

                objpoints.append(objp)
                imgpoints.append(corners)

        if len(imgpoints):
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            MyApp.get_calibration_error(ret, mtx, dist, rvecs, tvecs, imgpoints, objpoints)
            return ret, mtx, dist, rvecs, tvecs
        else:
            return None, None, None, None, None
    
    def camera_calibration_process(image_queue_R, result_queue_R, image_queue_L, result_queue_L, corners_queue_R, corners_queue_L, right_camera, left_camera, num_calib_im=6, n_row=7, n_col=9):
        """Subprocess function to constantly calibrate the camera."""
        calibration_images_R = []
        calibration_images_L = []
        while True:
            if right_camera:
                calibration_images_R = [image_queue_R.get() for i in range(num_calib_im)]
            if left_camera:
                calibration_images_L = [image_queue_L.get() for i in range(num_calib_im)]

            if any(elem is None for elem in calibration_images_R) or any(elem is None for elem in calibration_images_L):
                print("exiting from calibration")
                break

            # Example calibration process
            if right_camera:
                ret1, mtx1, dist1, rvecs1, tvecs1 = MyApp.calibrate_camera(calibration_images_R, corners_queue_R)
                result_queue_R.put((ret1, mtx1, dist1))  # Send results back
        
            # Example calibration process
            if left_camera:
                ret2, mtx2, dist2, rvecs2, tvecs2 = MyApp.calibrate_camera(calibration_images_L, corners_queue_L)
                result_queue_L.put((ret2, mtx2, dist2))  # Send results back

    # Détection des objets rouges dans une image
    # modifier, ajouter dilation pour suprimer la couleur dominee dans un objet (pour le considerer un seul objet)
    def detect_red_objects(video_feed, centers_queue):
        """
        Détecte des objets rouges dans une image et retourne leurs centres.

        Args:
        - frame : Image à analyser.
        - lower_red1, upper_red1 : Plage HSV pour le rouge vif.
        - lower_red2, upper_red2 : Plage HSV pour le rouge somb
    Returns:
        - frame : Image annotée avec des rectangles et points.
        - centers : Liste des centres des objets rouges détectés.
        """
        # Plages de couleur rouge (HSV)
        # lower_red1 = (0, 150, 100)
        # upper_red1 = (10, 255, 255)
        # lower_red2 = (170, 150, 100)
        # upper_red2 = (180, 255, 255)

        lower_red1 = np.array([95, 80, 60])
        upper_red1 = np.array([115, 255, 255])
        lower_red2 = np.array([95, 80, 60])
        upper_red2 = np.array([115, 255, 255])

        while True:
            ret, frame = video_feed.get()

            if frame is None:
                break
            else:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convertir en espace de couleur HSV
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)  # Masque pour rouge vif
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)  # Masque pour rouge sombre
                mask = cv2.bitwise_or(mask1, mask2)  # Combiner les deux masques

                # Nettoyer le masque pour éliminer le bruit
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                # Trouver les contours
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                centers = []

                # Parcourir les contours détectés
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # Filtrer les objets trop petits
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])  # Centre en X
                            cy = int(M["m01"] / M["m00"])  # Centre en Y
                            centers.append((cx, cy))  # Ajouter le centre à la liste
                            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # Dessiner un point bleu
                if len(centers):
                    x = centers[0][0]
                    y = centers[0][1]
                    centers_queue.put((centers[0], (x-10, y-10, x+10, y+10), 1))
            # sleep(0.1)

    def draw_predicted_shapes(self, frame, shape, content):
        if shape == 'points':
            x1, y1, x2, y2, label = content[0], content[1], content[2], content[3], content[4]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Draw object center
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
            # Add label
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if shape == 'line':
            last_position = [content[0], content[1]]
            predicted_position = [content[2], content[3]]
            # track_id = content[4]
            # if track_id not in self.object_paths.keys():
            #     self.object_paths.update({track_id: [predicted_position]})

            cv2.line(frame, (int(last_position[0]), int(last_position[1])),
                    (int(predicted_position[0]), int(predicted_position[1])),
                    (0, 0, 255), 2)  # Rouge pour prédiction
        return frame




print('creating app object')

def launch(app: MyApp, compute3d=True):

    # create empty spaces
    # object position (right camera)
    xy_right_cam = st.empty()
    # object position (left camera)
    xy_left_cam = st.empty()
    # object position (camera origin)
    xyz_cam_origin = st.empty()
    # distance from camera origin
    cam_origin_dist = st.empty()
    # object position (checkerboard origin)
    xyz_checkerboard_origin = st.empty()
    # distance from cherboard origin
    checkerboard_origin_dist = st.empty()
    # camera midpoint position (checkerboard origin)
    xyz_cam_midpoint = st.empty()
    # 
    # calibration matrix
    calib_mtx_right = st.empty()
    calib_mtx_left = st.empty()
    # distortion
    distortion_right = st.empty()
    distortion_left = st.empty()
    # intrinsic matrix
    # extrinsic matrix
    # calibration mean error
    # calib_error = st.empty()
    # 







    print('launching object detection app')

    app.compute3d = compute3d

    if app.calib:
        # Start the calibration subprocess
        print('starting camera calibration process')
        process = multiprocessing.Process(target=MyApp.camera_calibration_process, args=(app.image_queue_R, app.result_queue_R, app.image_queue_L, app.result_queue_L, app.corners_queue_R, app.corners_queue_L, app.right_camera, app.left_camera, app.num_calib_im, app.n_row, app.n_col))
        process.start()

    if app.detect:
        # prepare arguments
        w = int(app.cap_R.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(app.cap_R.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = app.cap_R.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
        frames = max(int(app.cap_R.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

        # start the object detection subprocesses / video_feed, centers_queue, w, h, fps, frames
        print('starting object detection processes')
        if app.right_camera:
            process_R = multiprocessing.Process(target=MyApp.detect_objects, args=(app.video_feed_R, app.centers_queue_R, w, h, fps, frames, app.method, app.model_path, app.conf_thresh, app.max_det, app.device))
            process_R.start()
        if app.left_camera:
            process_L = multiprocessing.Process(target=MyApp.detect_objects, args=(app.video_feed_L, app.centers_queue_L, w, h, fps, frames, app.method, app.model_path, app.conf_thresh, app.max_det, app.device))
            process_L.start()

        if app.predict:
            print('starting object prediction processes')
            if app.right_camera:
                process_Rp = multiprocessing.Process(target=MyApp.predict_objects, args=(app.detections_queue_R, app.predictions_queue_R, fps))
                process_Rp.start()
            if app.left_camera:
                process_Lp = multiprocessing.Process(target=MyApp.predict_objects, args=(app.detections_queue_L, app.predictions_queue_L, fps))
                process_Lp.start()

    i = 0
    ret_right, ret_left = None, None
    ret1, frame_right = None, None
    ret2, frame_left = None, None
    
    print('entering cycle')

    # Start the loop
    while True:

        i += 1
        # get image frames
        if app.right_camera:
            # print('getting right frame')
            ret1, frame_right = app.cap_R.read()
        if app.left_camera:
            ret2, frame_left = app.cap_L.read()
        # if no image then continue looping
        if not ret1 and not ret2:
            if cv2.waitKey(10)&0xFF == ord('q'):
                break
            continue


        # send image feed for detection
        if app.detect:
            if app.right_camera:
                app.video_feed_R.put((ret1, frame_right))
            if app.left_camera:
                app.video_feed_L.put((ret2, frame_left))
        
        # Send image feed for calibration
        if app.calib:
            if i % 10 == 0:
                # if app.image_queue_R.qsize() == 0:
                if app.right_camera:
                    app.image_queue_R.put(frame_right)
                # if app.image_queue_L.qsize() == 0:
                if app.left_camera:
                    app.image_queue_L.put(frame_left)


        # check if an object is detected
        if app.detect:
            try:
                # detect objects each camera returns
                # center : (x, y)
                # bbox: (x1, y1, x2, y2)
                # center = (((x1 + x2) / 2), ((y1 + y2) / 2))

                if app.right_camera and not app.centers_queue_R.empty():
                    app.centers_right, bbox_R, conf_R = app.centers_queue_R.get()
                    # print('centers right:', app.centers_right)
                    xy_right_cam.text(f"object (right camera)\n(X, Y) : ({app.centers_right[0]}, {app.centers_right[1]})")

                    # send detection to prediction
                    if app.predict:
                        app.detections_queue_R.put(((*bbox_R, conf_R, 0), frame_right))

                if app.left_camera and not app.centers_queue_L.empty():
                    app.centers_left, bbox_L, conf_L = app.centers_queue_L.get()
                    # print('centers left:', app.centers_left)
                    xy_left_cam.text(f"object (left camera)\n(X, Y) : ({app.centers_left[0]}, {app.centers_left[1]})")
                    
                    # send detection to prediction
                    if app.predict:
                        app.detections_queue_L.put(((*bbox_L, conf_L, 0), frame_left))

                if app.centers_right:
                    frame_right = cv2.circle(frame_right, app.centers_right, 10, (0, 0, 255), 1)
                    # cv2.putText(frame_right, str(float(conf_R)), app.centers_right, cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

                if app.centers_left:
                    frame_left = cv2.circle(frame_left, app.centers_left, 10, (0, 0, 255), 1)
                    # cv2.putText(frame_left, str(float(conf_L)), app.centers_left, cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

                if app.centers_left and app.centers_right:
                    app.uR, app.vR = app.centers_right  # First detected object in Caméra Droite
                    app.uL, app.vL = app.centers_left  # First detected object in Caméra Gauche

                # check if some predictions have been made
                if app.predict:
                    if app.right_camera and not app.predictions_queue_R.empty():
                        shape, content = app.predictions_queue_R.get()
                        frame_right = app.draw_predicted_shapes(frame_right, shape, content)
                    
                    if app.left_camera and not app.predictions_queue_L.empty():
                        shape, content = app.predictions_queue_L.get()
                        frame_left = app.draw_predicted_shapes(frame_left, shape, content)


            except ValueError as e:
                print(f"Error : {e}")

        # check if the camera has calibrated
        if app.calib:
            # Check for calibration results
            if app.right_camera and not app.result_queue_R.empty():
                ret1, mtx1, dist1 = app.result_queue_R.get()
                calib_mtx_right.text(f"calibration matrix (right camera)\n" + "\n".join([str(line) for line in mtx1]))
                distortion_right.text(f"distorion (right camera) : {dist1}")

                if ret1:
                    app.focal_length_x = mtx1[0, 0]  # Distance focale
                    app.cx = mtx1[0, 2]  # Coordonnée X du centre optique
                    app.cy = mtx1[1, 2]  # Coordonnée Y du centre optique

                    # print(f"Calibration Results:\nRet: {ret1}\nMatrix: {mtx1}\nDistortion: {dist1}")

                    # if we do not need constant calibration then we stop the calibration subprocess
                    if not app.constant_calib:
                        for i in range(10): # put 10 to make sure queue is read
                            if app.right_camera:
                                app.image_queue_R.put(None)  # Send exit signal to subprocess


            if app.left_camera and not app.result_queue_L.empty():
                ret2, mtx2, dist2 = app.result_queue_L.get()
                calib_mtx_left.text(f"calibration matrix (left camera)\n" + "\n".join([str(line) for line in mtx2]))
                distortion_left.text(f"distorion (left camera) : {dist2}")

                if ret2:
                    app.focal_length_x = mtx2[0, 0]  # Distance focale
                    app.cx = mtx2[0, 2]  # Coordonnée X du centre optique
                    app.cy = mtx2[1, 2]  # Coordonnée Y du centre optique

                    # print(f"Calibration Results:\nRet: {ret2}\nMatrix: {mtx2}\nDistortion: {dist2}")

                    # if we do not need constant calibration then we stop the calibration subprocess
                    if not app.constant_calib:
                        for i in range(10): # put 10 to make sure queue is read
                            if app.left_camera:
                                app.image_queue_L.put(None)  # Send exit signal to subprocess
            
            # get corners if chessboard corners detected
            if app.right_camera and not app.corners_queue_R.empty():
                ret_right, corners_right = app.corners_queue_R.get()
            if app.left_camera and not app.corners_queue_L.empty():
                ret_left, corners_left = app.corners_queue_L.get()

            if ret_right:
                frame_right = cv2.circle(frame_right, (int(corners_right[0][0][0]), int(corners_right[0][0][1])), 10, (0, 0, 0), 1)
                frame_right = cv2.drawChessboardCorners(frame_right, (app.n_row, app.n_col), corners_right, ret_right)
                app.origin_coords_right = (int(corners_right[0][0][0]), int(corners_right[0][0][1]))
            if ret_left:
                frame_left = cv2.circle(frame_left, (int(corners_left[0][0][0]), int(corners_left[0][0][1])), 10, (0, 0, 0), 1)
                frame_left = cv2.drawChessboardCorners(frame_left, (app.n_row, app.n_col), corners_left, ret_left)
                app.origin_coords_left = (int(corners_left[0][0][0]), int(corners_left[0][0][1]))
            
            if (ret_right and ret_left) or app.origin_cam_basis:
                app.origin_cam_basis = app.cam_2D_to_cam_3D(app.origin_coords_left[0], app.origin_coords_left[1], app.origin_coords_right[0], app.origin_coords_right[1])
                # print(f"3D position of origin (camera basis) : X={app.origin_cam_basis[0]}, Y={app.origin_cam_basis[1]}, Z={app.origin_cam_basis[2]}")

                app.cam_mid_point = app.cam_3D_to_real_3D(0, app.baseline/2, 0)
                print(f"3D position of camera (real world basis) : X={app.cam_mid_point[0]}, Y={app.cam_mid_point[1]}, Z={app.cam_mid_point[2]}")
                xyz_cam_midpoint.text(f"cameras mid point (checkerboard origin)\n(X, Y, Z) : ({app.cam_mid_point[0]}, {app.cam_mid_point[1]}, {app.cam_mid_point[2]})")

                should_be_000 = app.cam_3D_to_real_3D(app.origin_cam_basis[0], app.origin_cam_basis[1], app.origin_cam_basis[2])
                print(f"3D position of 0_0_0_ (real world basis) : X={should_be_000[0]}, Y={should_be_000[1]}, Z={should_be_000[2]}")

                cv2.putText(frame_right, f"X={should_be_000[0]}, Y={should_be_000[1]}, Z={should_be_000[2]}", (int(corners_right[0][0][0]), int(corners_right[0][0][1])), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame_left, f"X={should_be_000[0]}, Y={should_be_000[1]}, Z={should_be_000[2]}", (int(corners_left[0][0][0]), int(corners_left[0][0][1])), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)


            # compute 3d position if everything is available
            if app.detect and app.compute3d:
                try:
                    if app.uL and app.uR and app.vL and app.focal_length_x and app.baseline and app.cx and app.cy:
                        # returns x, y, z position with basis being the left camera
                        app.X_cam, app.Y_cam, app.Z_cam = app.cam_2D_to_cam_3D(app.uL, app.vL, app.uR)
                        xyz_cam_origin.text(f"object position (camera origin)\n(X, Y, Z) : ({app.X_cam}, {app.Y_cam}, {app.Z_cam})")
                        app.distance_cam = math.sqrt(app.X_cam*app.X_cam + app.Y_cam*app.Y_cam + app.Z_cam*app.Z_cam)
                        cam_origin_dist.text(f"distance of object from camera origin\ndist = {app.distance_cam}")

                        app.X_real, app.Y_real, app.Z_real = app.cam_3D_to_real_3D(app.X_cam, app.Y_cam, app.Z_cam)
                        xyz_checkerboard_origin.text(f"object position (checkerboard origin)\n(X, Y, Z) : ({app.X_real}, {app.Y_real}, {app.Z_real})")
                        app.distance_checker = math.sqrt(app.X_real*app.X_real + app.Y_real*app.Y_real + app.Z_real*app.Z_real)
                        checkerboard_origin_dist.text(f"distance of object from checkerboard origin\ndist = {app.distance_checker}")

                        # print(f"3D position of object (camera basis) : X={app.X_cam:.2f}, Y={app.Y_cam:.2f}, Z={app.Z_cam:.2f}")
                        print(f"3D position of object (real world basis) : X={app.X_real}, Y={app.Y_real}, Z={app.Z_real}")

                        cv2.putText(frame_right, f"X={app.X_real}, Y={app.Y_real}, Z={app.Z_real}", app.centers_right, cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(frame_left, f"X={app.X_real}, Y={app.Y_real}, Z={app.Z_real}", app.centers_left, cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

                except ValueError as e:
                    print(f"Error calculating 3D position: {e}")
        
        # Show the camera feed
        if app.right_camera:
            cv2.imshow("Camera1", frame_right)
        if app.left_camera:
            cv2.imshow("Camera2", frame_left)

        # Exit condition
        if cv2.waitKey(100)&0xFF == ord('q'):
            break

    print('-----------clean up before shutting down')
    # Cleanup
    app.cap_R.release()
    app.cap_L.release()
    cv2.destroyAllWindows()
    if app.calib:
        for i in range(10): # put 10 to make sure queue is read
            if app.right_camera:
                app.image_queue_R.put(None)  # Send exit signal to subprocess
            if app.left_camera:
                app.image_queue_L.put(None)  # Send exit signal to subprocess
    if app.detect:
        if app.right_camera:    
            app.video_feed_R.put((None, None))  # Send exit signal to subprocess
        if app.left_camera:
            app.video_feed_L.put((None, None))  # Send exit signal to subprocess

    if app.calib:
        process.join(1)
        if process.is_alive():
            # process.terminate()
            process.close()

    if app.detect:
        process_R.join(3)
        if app.right_camera and process_R.is_alive():
            # process_R.terminate()
            process_R.close()
        process_L.join(3)
        if app.left_camera and process_L.is_alive():
            # process_L.terminate()
            process_L.close()
        if app.predict:
            process_Rp.join(3)
            if app.right_camera and process_Rp.is_alive():
                # process_Rp.terminate()
                process_Rp.close()
            process_Lp.join(3)
            if app.left_camera and process_Lp.is_alive():
                # process_Lp.terminate()
                process_Lp.close()

# to update, detection in launch() expects only one value to be returned, fix it

# if __name__ == "__main__":
#     app = MyApp()
#     launch(app, calibrate=True, detect=True, detection_method='color_detector')
#     print('\n\nDONE\n\n')

