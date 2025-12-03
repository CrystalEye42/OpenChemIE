# askcos2_core

The core FastAPI app (a.k.a. the API gateway) for ASKCOSv2. The functional modules (a.k.a. the backend services), as well as the frontend, have their own repositories, which will be automatically pulled during setup based on the provided config file. Please refer to the [ASKCOS wiki](https://gitlab.com/mlpds_mit/askcosv2/askcos-docs/-/wikis/home) for more information. The README here only provides a quick start with full deployment.

## Quick Start

### Hardware

- **CPU**: x86, \>= 4 cores. Mac with ARM CPUs (Apple Silicon) are currently not supported.
- **RAM**: \>= 32 GB.
- **GPU** (only required for model retraining): \>= 8 GB GPU RAM.

### Software

- **OS**: Ubuntu (>= 20.04) recommended.
- **docker** (\>= version 20). See [Install docker](https://docs.docker.com/engine/install/).
- **CUDA** (only required for model retraining): \>= 11.3.
- **python**: >= 3.6 with `pyyaml` and `requests` (both pip installable)
- **git** and **make**.
- **ssh access** set up for gitlab.com. See [Add an SSH key to Gitlab](https://docs.gitlab.com/ee/user/ssh.html#add-an-ssh-key-to-your-gitlab-account).

Please note that \>= 32 GB of RAM is highly recommended for the full deployment of ASKCOS. If memory size is a constraint, please check out **04-Deployment** for customizing a local deployment (e.g., backend only, or minimalist deployment to use only the retrosynthesis functionality).

### Deployment commands

```shell
$ mkdir ASKCOSv2
$ cd ASKCOSv2
$ git clone git@gitlab.com:mlpds_mit/askcosv2/askcos2_core.git
$ cd askcos2_core
$ make deploy
```

The setup phase will then run automatically, e.g., cloning the repos for backend modules, building docker images, seeding databases. After setup, the web app will be started with the Graphical User Interface accessible at the host ip address, e.g., 0.0.0.0. Simply type this address in your browser, and the welcome page will appear. **CONTINUE AS GUEST** to explore ASKCOS. Signup or login is not required for using most modules.

The web app is written in [Vue.js](https://vuejs.org/), which communicates with the backend. The core piece of the backend, the _API Gateway_, is written in [FastAPI](https://fastapi.tiangolo.com/), with its own interactive documentation accessible at [http://0.0.0.0:9100/docs](http://0.0.0.0:9100/docs).
