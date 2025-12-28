import config_prod
# vers le dossier contenant les ressources
DATA_PATH = "..\\data"
# dataset original
DATASET_ORIGINAL_PATH = "..\\data\\dataset_birds_original"
# dataset équilibré mais non détouré avec 523 classes
DATASET_CLEAN_PATH = "..\\data\\dataset_birds_clean"
# dataset équilibré et détouré avec 523 classes
DATASET_CLEAN_WO_BACKGROUND_PATH = "..\\data\\dataset_birds_clean_wo_background"
# dataset équilibré et détouré avec 395 classes
DATASET_CLEAN_WO_BACKGROUND_REDUIT = "..\\data\\dataset_birds_clean_reduit_wo_background"
# dataset non équilibré et détouré, regroupant les oiseaux par familles
DATASET_CLEAN_WO_BACKGROUND_FAMILY = "..\\data\\dataset_birds_clean_wo_background_family"
# dataset non équilibré et détouré, regroupant les oiseaux par ordres
DATASET_CLEAN_WO_BACKGROUND_ORDERS = "..\\data\\dataset_birds_clean_wo_background_orders"
# le dataset divisé en deux parties
DATASET_CLEAN_WO_BACKGROUND_HALF1 = "..\\data\\half_dataset_birds_clean_wo_background_part_1"
DATASET_CLEAN_WO_BACKGROUND_HALF2 = "..\\data\\half_dataset_birds_clean_wo_background_part_2"

# dataset test
DATASET_TEST_PATH = "..\\data\\dataset_birds_test"
# dataset test détouré
DATASET_TEST_WO_BACKGROUND_PATH = "..\\data\\dataset_birds_test_wo_background"

# chemin vers les modèles de Deep Learning
MODELS_PATH = "..\\data\\models_finaux"
# chemin utilisé pour les modèles de Machine Learning
DATA_ML_PATH = "..\\data\\data_ML"

# chemin vers le projet pour faire un map de profondeur
DEPTH_ANYTHING = "..\\data\\Depth-Anything" 
# liens relatifs à la position de Depth-Anything, et non pas des notebooks
DEPTH_ANYTHING_INDIR = "..\\dataset_birds_wo_background\\temp"
DEPTH_ANYTHING_OUTDIR = "..\\dataset_birds_wo_background\\temp_depth"

 # lien facultatif mais peut aider, à modifier en fonction de votre venv
PYTHON_VENV_EXECUTABLE = "..\\..\\birdEnvOlder\\Scripts\\python.exe"