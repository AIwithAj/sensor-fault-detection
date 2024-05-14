import json
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.entity.artifact_entity import DataValidationArtifact
from sensor.logger import logging

STAGE_NAME = "Data Transformation stage"

if __name__ == '__main__':
    try:
        TrainPipeline.is_pipeline_running=True
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        # Deserialize data_validation_artifact from JSON
        with open('data_validation_artifact.json', 'r') as json_file:
            validation_dict = json.load(json_file)
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_dict['validation_status'],
                valid_train_file_path=validation_dict['valid_train_file_path'],
                valid_test_file_path=validation_dict['valid_test_file_path'],
                invalid_train_file_path=validation_dict['invalid_train_file_path'],
                invalid_test_file_path=validation_dict['invalid_test_file_path'],
                drift_report_file_path=validation_dict['drift_report_file_path']
            )

        obj = TrainPipeline()
        data_transformation_artifact = obj.start_data_transformation(data_validation_artifact=data_validation_artifact)

        # Serialize data_transformation_artifact to JSON
        transformation_dict = {
            'transformed_object_file_path': data_transformation_artifact.transformed_object_file_path,
            'transformed_train_file_path': data_transformation_artifact.transformed_train_file_path,
            'transformed_test_file_path': data_transformation_artifact.transformed_test_file_path
        }

        with open('data_transformation_artifact.json', 'w') as json_file:
            json.dump(transformation_dict, json_file)

        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
