import json
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.entity.artifact_entity import DataIngestionArtifact
from sensor.logger import logging

STAGE_NAME = "Data Ingestion stage"

if __name__ == '__main__':
    try:
        TrainPipeline.is_pipeline_running=True
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TrainPipeline()
        data_ingestion_artifact = obj.start_data_ingestion()

        # Serialize data_ingestion_artifact to JSON
        artifact_dict = {
            'trained_file_path': data_ingestion_artifact.trained_file_path,
            'test_file_path': data_ingestion_artifact.test_file_path
        }

        with open('data_ingestion_artifact.json', 'w') as json_file:
            json.dump(artifact_dict, json_file)

        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
