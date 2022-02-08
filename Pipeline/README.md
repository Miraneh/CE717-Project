# Install requirements

Before proceeding install libraries you'll need (mostly for training the model) as specified in `requirements.txt`
```
  pip install requirements.txt
```


# Build mlflow models
First you should run these to get the runner ids for next part.
```
  python pre_processing.py
  python models.py
```
These will print out the id for you.
# Make containers
Add those ids from previous part to the end of file to build. (You could change file name pattern for `pre_proccessing` and `models` in their respective python scripts)
```
  mlflow models build-docker -m mlflow_models/pre_processing_9f5fb7dd33914c1a8c9b0416da3d7ba6 -n processed --enable-mlserver
  mlflow models build-docker -m mlflow_models/models_e7fd64a377594f8eb23ee691d3637354 -n trained-model --enable-mlserver
```

# Run containers
Now we can run each image we just built.
```
  docker run -p 1234:8080 --name processed
  docker run -p 1235:8080 --name trained-model
```
