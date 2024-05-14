import json
import sys
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity.artifact_entity import ClassificationMetricArtifact
import os

STAGE_NAME = "Model Pusher stage"

if __name__ == '__main__':
    try:
        TrainPipeline.is_pipeline_running = True
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        # Load model_eval_artifact from JSON
        with open('model_eval_artifact.json', 'r') as json_file:
            eval_dict = json.load(json_file)
            model_eval_artifact = ModelEvaluationArtifact(
                is_model_accepted=eval_dict['is_model_accepted'],
                improved_accuracy=eval_dict['improved_accuracy'],
                best_model_path=eval_dict['best_model_path'],
                trained_model_path=eval_dict['trained_model_path'],
                train_model_metric_artifact=ClassificationMetricArtifact(**eval_dict['train_model_metric_artifact']),
                best_model_metric_artifact=ClassificationMetricArtifact(**eval_dict['best_model_metric_artifact'])
            )

        if not model_eval_artifact.is_model_accepted:
             logging.info("Trained model is not better than the best model")

        obj= TrainPipeline()
        model_pusher_artifact = obj.start_model_pusher(model_eval_artifact)

        TrainPipeline.is_pipeline_running = False
        # Sync artifact directory to S3
        obj.sync_artifact_dir_to_s3()
        # Sync saved model directory to S3
        obj.sync_saved_model_dir_to_s3()
# List of JSON files
        json_files = [
            'data_ingestion_artifact.json',
            'data_validation_artifact.json',
            'data_transformation_artifact.json',
            'model_trainer_artifact.json',
            'model_pusher_artifact.json'
        ]

        # Remove each JSON file
        logging.info("Removing json files ....")
        for json_file in json_files:
            if os.path.exists(json_file):
                os.remove(json_file)
                print(f"Removed: {json_file}")
            else:
                print(f"Not found: {json_file}")
        logging.info("successfully Removed json files")
        

        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        # Sync artifact directory to S3 before raising an exception
        obj.sync_artifact_dir_to_s3()
        TrainPipeline.is_pipeline_running = False
        raise SensorException(e, sys)
