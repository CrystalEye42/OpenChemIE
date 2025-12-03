# Contributing to ASKCOSv2
We want to make contributing to ASKCOSv2 as easy and transparent as
possible.

## Contributing to an existing repo
1. Fork the repo and create your branch from `dev`, or `main` if `dev` doesn't exist.
2. If you've added code that should be tested, add tests (which we gently enforce).
3. Ensure the test suite passes.
4. For askcos2_core and askcos2_site only, make sure your code lints.
5. Submit the merge request into the forked branch (`dev` or `main`).

## Contributing a new module

[//]: # (The core requirement is a `deploy.yaml` file in your repo which specifies at least two commands, one for building the Docker image &#40;GPU and/or CPU&#41; and the other for starting the prediction service in Docker. This file will be read during the deployment process to start your prediction service, and also used as a reference by the ASKCOS team for adding needed infrastructural support for your module.)

### Step 1: wrap your prediction code as a microservice

- Option 1 (basic): using Flask/FastAPI

We will use the <a href="https://gitlab.com/mlpds_mit/askcosv2/site_selectivity/-/tree/dev/">`site_selectivity`</a> module as a reference. Assuming you have some existing code that can run a prediction task, for example, <a href="https://gitlab.com/mlpds_mit/askcosv2/site_selectivity/-/blob/dev/site_selectivity.py#L115">`SiteSelectivityModel().predict()`</a>.
The first step is wrapping this function as an API endpoint, with the framework of your choice (we recommend FastAPI). See the <a href="https://gitlab.com/mlpds_mit/askcosv2/site_selectivity/-/blob/dev/site_selectivity_server.py#L39-L63">relevant code section</a> in `site_selectivity_server.py` as an example. This turns your prediction code into a service that can be run with
```
python your_server.py
```
and can be queried once started.

- Option 2 (advanced): using Torchserve 

Serving with Torchserve requires a few more steps, but makes it trivially easy to add new models later on. We will use the <a href="https://gitlab.com/mlpds_mit/askcosv2/forward_predictor/augmented_transformer">`augmented_transformer`</a> module as a reference. Assuming you have some existing code that can run a prediction task, for example, `ServerModel().run()`.
The first step is wrapping this function in `your_handler.py` under `YourHandler.inference()`. See the <a href="https://gitlab.com/mlpds_mit/askcosv2/forward_predictor/augmented_transformer/-/blob/main/at_handler.py#L67-L69">relevant code section</a> in `at_handler.py` as an example.

Once the handler has been implemented, you need to package your code, including the handler, with other required files (e.g., model checkpoints, data files) into a *servable model archive* using `torch-model-archiver`, similar to <a href="https://gitlab.com/mlpds_mit/askcosv2/forward_predictor/augmented_transformer/-/blob/main/scripts/archive_in_docker.sh">how it was done for augmented Transformer</a>. Note that while we typically perform this archiving step in Docker, this is not required and you can also do it in a conda environment. If successful, you should be able to obtain a model archive ending with `.mar`, which can then be served by
```
torchserve \
  --start \
  --foreground \
  --ncs \
  --model-store=${FOLDER_CONTAINING_MARS} \
  --models \
  ${MODEL_NAME}=${MODEL_NAME}.mar
```
and can be queried once started.

### Step 2: containerized your microservice in Docker
Containerizing this service in Docker is then straightforward if you already have some conda environment. Simply create a Dockerfile similar to the sample Dockerfiles for <a href="https://gitlab.com/mlpds_mit/askcosv2/site_selectivity/-/blob/dev/Dockerfile_site_cpu">site_selectivity</a>, or for <a href="https://gitlab.com/mlpds_mit/askcosv2/forward_predictor/augmented_transformer/-/blob/main/Dockerfile_cpu">augmented_transformer</a> to install the dependencies. Then build the Docker image with
```
export ASKCOS_REGISTRY=registry.gitlab.com/mlpds_mit/askcosv2
docker build -f your_Dockerfile -t ${ASKCOS_REGISTRY}/your_module:1.0-cpu .
```
after which the containerized microservice should be runnable with a single command, e.g.,
```
docker run --rm -p 9601:9601 -t ${ASKCOS_REGISTRY}/your_module:1.0-cpu
```
Or in the case of lengthier command, organize it into a start script similar to <a href="https://gitlab.com/mlpds_mit/askcosv2/forward_predictor/augmented_transformer/-/blob/main/scripts/serve_cpu_in_docker.sh">`scripts/serve_cpu_in_docker.sh`</a> for augmented_transformer, which can then be run by
```
sh scripts/serve_cpu_in_docker.sh
```

### Step 3 (optional): prepare `deployment.yaml`
<span style="color:red">TODO</span>


### Step 4: contact us
Once Steps 1-3 have been completed, contact the ASKCOS team (mlpds_support@mit.edu) to initiate the integration of your service into ASKCOSv2. What follows would mostly be bookkeeping, such as adding wrapper that calls your service at the API gateway (in `askcos2_core`). Since we may be shifting data (to our cloud storage), reassigning ports, and adding additional supports (e.g., singularity and async task management), we would prefer to handle these hassles for you.

## Contributing a new model for an existing module, or contributing new data (e.g., reactions, buyables).
Please contact the ASKCOS team directly (mlpds_support@mit.edu).

## Issues

We use GitLab issues to track public bugs. Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to ASKCOSv2, you agree that your contributions will be licensed under the LICENSE file in the root directory of the repo, if there is one.
