
import streamlit as st
from part3_script import MyApp, launch


def lancer(right_camera, right_source, left_camera, left_source, calib, constant_calib, detect, method, model_path, conf_thresh, max_det, device, camera_dist, predict):
    st.write("application lancé avec succes")
    app = MyApp(right_camera, right_source, left_camera, left_source, calib, constant_calib, detect, method, model_path, conf_thresh, max_det, device, camera_dist, predict)
    # calibrate :bool , detection_method= 'yolo' ou 'color_detector'
    launch(app)
    print('\n\nDONE\n\n')

if __name__ == "__main__":


    # Titre principal de l'application
    st.title("Affichage des résultats")

    # object position (right camera)
    # object position (left camera)
    # object position (camera origin)
    # object position (checkboard origin)
    # camera midpoint position (checkboard origin)
    # 
    # calibration matrix
    # distortion
    # intrinsic matrix
    # extrinsic matrix
    # calibration mean error
    # 

    # # Créer un espace vide
    # err_calibrage = st.empty()
    # # Afficher du texte dans cet espace
    # err_calibrage.text("err_calibrage")

    # xyz_screen = st.empty()
    # xyz_screen.text("xyz_screen")

    # xyz_camera = st.empty()
    # xyz_camera.text("xyz_camera")

    # =================== Créer une barre latérale ========================
    # to update, detection in launch() expects only one value to be returned, fix it
    st.sidebar.title("Espace de controle de l'application")

    right_camera = st.sidebar.checkbox("Right camera active", True)
    right_source = st.sidebar.text_input("Source Path:", "1")  # Default to pc webcam
    left_camera = st.sidebar.checkbox("Left camera active", True)
    left_source = st.sidebar.text_input("Source Path:", "2")  # Default to first phone cam
    calib = st.sidebar.checkbox("calibration active", True)
    constant_calib = st.sidebar.checkbox("constant calibration active", True)

    # Widget selectbox (color_detector is for testing since yolo takes too long)
    detect = st.sidebar.checkbox("object detection active", True)
    method = st.sidebar.selectbox("choose detection method :", ["yolo", "color_detector"])

    # Input fields
    model_path = st.sidebar.text_input("Model Path:", "best.pt")
    conf_thresh = st.sidebar.slider("Confidence Threshold:", 0.1, 1.0, 0.7)
    max_det = st.sidebar.number_input("Max Detections:", min_value=1, max_value=500, value=1)
    device = st.sidebar.selectbox("Device:", ["cpu", "cuda"])
    camera_dist = st.sidebar.number_input("distance between cameras (mm):", value=100)

    # prediction (new option)
    predict = st.sidebar.checkbox("object prediction active", False)


    # Liste d'options

    # Ajouter des boutons à la barre latérale
    if st.sidebar.button("Lancer Le modele choisi"):
        lancer(right_camera, int(right_source), left_camera, int(left_source), calib, constant_calib, detect, method, model_path, conf_thresh, max_det, device, camera_dist, predict)


