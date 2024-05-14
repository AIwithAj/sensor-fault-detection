import json
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.entity.artifact_entity import DataValidationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, ClassificationMetricArtifact
from sensor.logger import logging

STAGE_NAME = "Model Evaluation stage"

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

        # Deserialize model_trainer_artifact from JSON
        with open('model_trainer_artifact.json', 'r') as json_file:
            trainer_dict = json.load(json_file)
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=trainer_dict['trained_model_file_path'],
                train_metric_artifact=ClassificationMetricArtifact(**trainer_dict['train_metric_artifact']),
                test_metric_artifact=ClassificationMetricArtifact(**trainer_dict['test_metric_artifact'])
            )

        obj = TrainPipeline()
        model_eval_artifact = obj.start_model_evaluation(data_validation_artifact=data_validation_artifact, model_trainer_artifact=model_trainer_artifact)

        # Convert model_eval_artifact to a dictionary
        model_eval_dict = {
            'is_model_accepted': model_eval_artifact.is_model_accepted,
            'improved_accuracy': model_eval_artifact.improved_accuracy,
            'best_model_path': model_eval_artifact.best_model_path,
            'trained_model_path': model_eval_artifact.trained_model_path,
            'train_model_metric_artifact': {
                'f1_score': model_eval_artifact.train_model_metric_artifact.f1_score,
                'precision_score': model_eval_artifact.train_model_metric_artifact.precision_score,
                'recall_score': model_eval_artifact.train_model_metric_artifact.recall_score
            },
            'best_model_metric_artifact': {
                'f1_score': model_eval_artifact.best_model_metric_artifact.f1_score,
                'precision_score': model_eval_artifact.best_model_metric_artifact.precision_score,
                'recall_score': model_eval_artifact.best_model_metric_artifact.recall_score
            }
        }

        # Serialize model_eval_artifact to JSON
        with open('model_eval_artifact.json', 'w') as json_file:
            json.dump(model_eval_dict, json_file, indent=4)

        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
