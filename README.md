# drake_6881
Run `python kuka_pydrake_sim_no_cam.py` after opening meshcat in a terminal.

## Building and running the docker container
First run:

    docker build -t move_until_contact . --build-arg DRAKE_URL=https://drake-packages.csail.mit.edu/drake/nightly/drake-latest-xenial.tar.gz

Then run:

    docker run -it -p 8080:8080 -p 7000:7000 --rm move_until_contact

This will open a docker container with meschat running the background. Pressing enter will open a new terminal line, where you can run

    cd /test_dir && python kuka_pydrake_sim_no_cam.py
    
This will start the simulation the meshcat.
