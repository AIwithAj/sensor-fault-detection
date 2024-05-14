import json
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from sensor.logger import logging

STAGE_NAME = "Model Training stage"

if __name__ == '__main__':
    try:
        TrainPipeline.is_pipeline_running = True
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        # Deserialize data_transformation_artifact from JSON
        with open('data_transformation_artifact.json', 'r') as json_file:
            transformation_dict = json.load(json_file)
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=transformation_dict['transformed_object_file_path'],
                transformed_train_file_path=transformation_dict['transformed_train_file_path'],
                transformed_test_file_path=transformation_dict['transformed_test_file_path']
            )

        obj = TrainPipeline()
        model_trainer_artifact = obj.start_model_trainer(data_transformation_artifact)
        # example model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path='artifact\\05_14_2024_19_39_32\\model_trainer\\trained_model\\model.pkl', train_metric_artifact=ClassificationMetricArtifact(f1_score=1.0, precision_score=1.0, recall_score=1.0), test_metric_artifact=ClassificationMetricArtifact(f1_score=0.9855891085523933, precision_score=0.994301578024547, recall_score=0.9770279971284996))

        # Serialize model_trainer_artifact to JSON
        artifact_dict = {
            'trained_model_file_path': model_trainer_artifact.trained_model_file_path,
            'train_metric_artifact': {
                'f1_score': model_trainer_artifact.train_metric_artifact.f1_score,
                'precision_score': model_trainer_artifact.train_metric_artifact.precision_score,
                'recall_score': model_trainer_artifact.train_metric_artifact.recall_score
            },
            'test_metric_artifact': {
                'f1_score': model_trainer_artifact.test_metric_artifact.f1_score,
                'precision_score': model_trainer_artifact.test_metric_artifact.precision_score,
                'recall_score': model_trainer_artifact.test_metric_artifact.recall_score
            }
        }

        with open('model_trainer_artifact.json', 'w') as json_file:
            json.dump(artifact_dict, json_file)

        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
