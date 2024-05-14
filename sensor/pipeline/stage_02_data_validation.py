import json
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from sensor.logger import logging

STAGE_NAME = "Data Validation stage"

if __name__ == '__main__':
    try:
        TrainPipeline.is_pipeline_running=True
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        # Deserialize data_ingestion_artifact from JSON
        with open('data_ingestion_artifact.json', 'r') as json_file:
            artifact_dict = json.load(json_file)
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=artifact_dict['trained_file_path'],
                test_file_path=artifact_dict['test_file_path']
            )

        obj = TrainPipeline()
        data_validation_artifact = obj.start_data_validaton(data_ingestion_artifact=data_ingestion_artifact)

        # Serialize data_validation_artifact to JSON
        validation_dict = {
            'validation_status': data_validation_artifact.validation_status,
            'valid_train_file_path': data_validation_artifact.valid_train_file_path,
            'valid_test_file_path': data_validation_artifact.valid_test_file_path,
            'invalid_train_file_path': data_validation_artifact.invalid_train_file_path,
            'invalid_test_file_path': data_validation_artifact.invalid_test_file_path,
            'drift_report_file_path': data_validation_artifact.drift_report_file_path
        }

        with open('data_validation_artifact.json', 'w') as json_file:
            json.dump(validation_dict, json_file)

        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
