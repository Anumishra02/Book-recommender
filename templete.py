import logging
logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s:')
project_name="books_recommender"

list_of_files=[
    f"{project_name}/_init_.py",
    f"{project_name}/components/_init_.py",
    f"{project_name}/components/stage_00_data_ingestion.py",
    f"{project_name}/components/stage_01_data_validation.py",
    f"{project_name}/components/stage_02_data_transformation.py",
    f"{project_name}/components/stage_03_model_trainer.py",
    f"{project_name}/config/_init_.py",
    f"{project_name}/config/configuration.py",
    f"{project_name}/constant/_init_.py",
    f"{project_name}/entity/_init_.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/exception/_init_.py",
    f"{project_name}/exception/exception_handler.py",
    f"{project_name}/logger/_init_.py",
    f"{project_name}/logger/log.py",
    f"{project_name}/pipeline/_init_.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/utils/_init_.py",
    f"{project_name}/utils/utils.py",
    ".dockerignore",
    "app.py",
    "Dockerfile",
    "setup.py"
    "test.py"



]