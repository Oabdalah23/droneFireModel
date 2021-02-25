from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="hololens")
trainer.setTrainConfig(object_names_array=["fire", "smoke"], batch_size=4, num_experiments=5, train_from_pretrained_model="detection_model-ex-002--loss-0037.097.h5")
trainer.trainModel()